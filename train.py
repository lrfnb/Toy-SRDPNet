import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from timm.scheduler import MultiStepLRScheduler, CosineLRScheduler
import datasets
import models
import utils
from test import eval_psnr
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from models.resolution_regression_net import ResolutionRegressionNet
import torch.nn.functional as F
import random
import numpy as np
import math

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def calculate_loss(pixel_loss, perceptual_loss, target_ratio=0.1, eps=1e-12, clamp_range=(1e-3, 1e3)):
    with torch.no_grad():
        Lp = pixel_loss.detach().clone()
        Lv = perceptual_loss.detach().clone()

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(Lp, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(Lv, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        Lp = Lp / world_size
        Lv = Lv / world_size

    w = (target_ratio / (1 - target_ratio)) * (Lp / (Lv + eps))
    if clamp_range is not None:
        w = torch.clamp(w, clamp_range[0], clamp_range[1])

    weighted_perceptual_loss = w * perceptual_loss
    loss = pixel_loss + weighted_perceptual_loss
    return loss, weighted_perceptual_loss, w


def calculate_perceptual_loss(pred, gt, diffusion):
    pred = pred.float()
    gt = gt.float()

    b = pred.shape[0]
    t = torch.randint(0, 1000, (1,), device=pred.device).long().expand(b)

    with torch.cuda.amp.autocast(enabled=False):  
        x_T = diffusion.q_sample(gt, t)
        xwave_T = diffusion.q_sample(pred, t)
        model_mean_gt, model_mean_hr = diffusion.perceptual_loss(x_T, xwave_T, t)
        
    l1_loss = 0.0
    for p, g in zip(model_mean_gt, model_mean_hr):
        l1_loss += F.l1_loss(p.float(), g.float())

    return l1_loss


def load_diffusion_model(device):
    data = torch.load('./results/model-50.pt', map_location=device)
    UNet = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    ).to(device)
    freeze_model(UNet)
    model = GaussianDiffusion(
        UNet, image_size=128, timesteps=1000).to(device)
    model.load_state_dict(data['model'])
    freeze_model(model)
    return model

def load_regression_model():
    model = ResolutionRegressionNet()
    model_path = '/data2/lrf/HIIF/results/best_loss_net_R.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace("module.", "")] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    freeze_model(model)
    return model

def make_scheduler(optimizer, config, epoch_start):
    # Original
    if config.get('multi_step_lr') is not None:
        print('multi_step_lr with config', config['multi_step_lr'])
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        if epoch_start is not None:
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
                # timm's multi step lr
    elif config.get('multi_step_lr_warmup') is not None:
        print('multi_step_lr_warmup with config', config['multi_step_lr_warmup'])
        lr_scheduler = MultiStepLRScheduler(optimizer, **config['multi_step_lr_warmup'])
        if epoch_start is not None:
            lr_scheduler.step(epoch=epoch_start - 1)
            # timm's cosine lr
    elif config.get('cosine_lr_warmup') is not None:
        print('cosine_lr_warmup with config', config['cosine_lr_warmup'])
        lr_scheduler = CosineLRScheduler(optimizer, **config['cosine_lr_warmup'], **{'t_initial': config['epoch_max']})
        if epoch_start is not None:
            for _ in range(epoch_start - 1):
                lr_scheduler.step(epoch=epoch_start - 1)
    else:
        lr_scheduler = None
    return lr_scheduler


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    resume_path = config.get('resume')

    if resume_path and os.path.exists(resume_path):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    lr_scheduler = make_scheduler(optimizer, config, epoch_start)
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    log('optimizer: {}'.format(optimizer))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, epoch):
    diffusion = load_diffusion_model(device="cuda")
    regression_model = load_regression_model()
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()

    pixel_loss_tracker = utils.Averager()
    perceptual_loss_raw_tracker = utils.Averager()  
    perceptual_loss_weighted_tracker = utils.Averager()
    weight_tracker = utils.Averager() 
    pixel_ratio_tracker = utils.Averager()
    perceptual_ratio_tracker = utils.Averager()

    metric_fn = utils.calc_psnr

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    num_dataset = 800  
    iter_per_epoch = int(
        num_dataset / config.get('train_dataset')['batch_size'] * config.get('train_dataset')['dataset']['args'][
            'repeat'])
    iteration = 0

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        _,regression_features = regression_model(batch['regressions'])
        #print(f"this is regression_feature:{regression_features.shape}") #[b,256,1]
        # pred = model(inp, batch['coord'], batch['cell'], batch['style_image'])
        pred = model(inp, batch['coord'], batch['cell'], batch['style_image'],regression_features)
        gt = (batch['gt'] - gt_sub) / gt_div
        pixel_loss = loss_fn(pred, gt)
        perceptual_loss_raw = calculate_perceptual_loss(pred, gt, diffusion)


        loss, perceptual_loss_weighted, weight = calculate_loss(pixel_loss, perceptual_loss_raw)

        pixel_loss_val = pixel_loss.item()
        perceptual_loss_raw_val = perceptual_loss_raw.item()
        perceptual_loss_weighted_val = perceptual_loss_weighted.item()
        weight_val = weight.item()
        total_loss_val = loss.item()

        if total_loss_val > 0:
            pixel_ratio = (pixel_loss_val / total_loss_val) * 100
            perceptual_ratio = (perceptual_loss_weighted_val / total_loss_val) * 100
        else:
            pixel_ratio = 0
            perceptual_ratio = 0


        pixel_loss_tracker.add(pixel_loss_val)
        perceptual_loss_raw_tracker.add(perceptual_loss_raw_val)
        perceptual_loss_weighted_tracker.add(perceptual_loss_weighted_val)
        weight_tracker.add(weight_val)
        pixel_ratio_tracker.add(pixel_ratio)
        perceptual_ratio_tracker.add(perceptual_ratio)

        psnr = metric_fn(pred, gt)


        writer.add_scalars('loss', {'train': loss.item()}, (epoch - 1) * iter_per_epoch + iteration)
        writer.add_scalars('psnr', {'train': psnr}, (epoch - 1) * iter_per_epoch + iteration)


        writer.add_scalars('loss_components', {
            'pixel_loss': pixel_loss_val,
            'perceptual_loss_raw': perceptual_loss_raw_val,
            'perceptual_loss_weighted': perceptual_loss_weighted_val
        }, (epoch - 1) * iter_per_epoch + iteration)

        writer.add_scalars('loss_ratios', {
            'pixel_ratio': pixel_ratio,
            'perceptual_ratio': perceptual_ratio
        }, (epoch - 1) * iter_per_epoch + iteration)

        writer.add_scalar('perceptual_weight', weight_val, (epoch - 1) * iter_per_epoch + iteration)

        iteration += 1

        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None
        loss = None


    return (train_loss.item(),
            pixel_loss_tracker.item(),
            perceptual_loss_raw_tracker.item(),
            perceptual_loss_weighted_tracker.item(),
            weight_tracker.item(),
            pixel_ratio_tracker.item(),
            perceptual_ratio_tracker.item())


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_results = train(train_loader, model, optimizer, epoch)
        (train_loss, avg_pixel_loss, avg_perceptual_loss_raw, avg_perceptual_loss_weighted,
         avg_weight, avg_pixel_ratio, avg_perceptual_ratio) = train_results

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, MultiStepLR):
                lr_scheduler.step()
            else:
                lr_scheduler.step(epoch=epoch)

        log_info.append('train: loss={:.4f} lr={:.4f}'.format(train_loss, optimizer.param_groups[0]['lr']))

        loss_breakdown = 'pixel_loss={:.4f}({:.1f}%) perceptual_loss_raw={:.4f} perceptual_loss_weighted={:.4f}({:.1f}%) weight={:.4f}'.format(
            avg_pixel_loss, avg_pixel_ratio, avg_perceptual_loss_raw, avg_perceptual_loss_weighted,
            avg_perceptual_ratio, avg_weight)
        log_info.append('loss_breakdown: ' + loss_breakdown)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                                data_norm=config['data_norm'],
                                eval_type=config.get('eval_type'),
                                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
