import argparse
import os
import math
from functools import partial
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.resolution_regression_net import ResolutionRegressionNet
import datasets
import models
import utils
from time import time

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

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

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


# for fast
def batched_predict_fast(model, inp, coord, cell, style_image,regression_feature,bsize):
    with torch.no_grad():
        regression_model = load_regression_model()
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        regressions = regression_model(regression_feature)
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord=coord[:, ql: qr, :], cell=cell,style_image=style_image,regression_feature=regressions)
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=2)
    return pred

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=True,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div

        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]

            coord = utils.make_coord((scale * (h_old + h_pad), scale * (w_old + w_pad)), flatten=False).unsqueeze(0).cuda()
            cell = torch.ones_like(batch['cell'])
            cell[:, 0] *= 2 / inp.shape[-2] / scale
            cell[:, 1] *= 2 / inp.shape[-1] / scale

        else:
            h_pad = 0
            w_pad = 0
            coord = batch['coord']
            cell = batch['cell']

        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell)
        else:
            if fast:
                pred = batched_predict_fast(model, inp, coord, cell * max(scale / scale_max, 1),
                                       batch['style_image'],batch['regressions'],eval_bsize)
            else:
                pred = batched_predict(model, inp, coord, cell * max(scale / scale_max, 1),
                                       batch['style_image'],batch['regressions'],eval_bsize)  # cell clip for extrapolation

        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None and window_size != 0:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1]*batch['coord'].shape[2] / (ih * iw))
            shape = [batch['inp'].shape[0], 3, round(ih * s), round(iw * s)]
            batch['gt'] = batch['gt'].view(*shape).contiguous()

            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1]*coord.shape[2] / (ih * iw))
            shape = [batch['inp'].shape[0], 3, round(ih * s), round(iw * s)]
            pred = pred.view(*shape).contiguous()
            pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]

        res = metric_fn(pred, batch['gt'])
        val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=0, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    model = torch.compile(model)
    start_time = time()
    res = eval_psnr(loader, model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    window_size=int(args.window),
                    scale_max=int(args.scale_max),
                    fast=args.fast,
                    verbose=True)
    end_time = time()
    print('result: {:.4f}'.format(res))
    print('Elapsed time: {:.2f} seconds'.format(end_time - start_time))