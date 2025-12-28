import argparse
import os
from PIL import Image
import random
import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict_fast
from collections import OrderedDict


def bicubic_resample(image, scale):
    return image.resize((int(image.width * scale), int(image.height * scale)), Image.BICUBIC)


def process_image(input_path, output_path, model, style_image, scale, scale_max=4):
    img = Image.open(input_path).convert('RGB')

    img = transforms.ToTensor()(img)

    h = int(img.shape[-2] * scale)
    w = int(img.shape[-1] * scale)

    coord = make_coord((h, w), flatten=False).unsqueeze(0).cuda()
    cell = torch.tensor([2 / h, 2 / w], dtype=torch.float32).unsqueeze(0)

    inp = (img - 0.5) / 0.5
    _, h_old, w_old = inp.size()

    if args.window_size != 0:

        h_pad = (h_old // args.window_size + 1) * args.window_size - h_old
        w_pad = (w_old // args.window_size + 1) * args.window_size - w_old
        inp = torch.cat([inp, torch.flip(inp, [1])], 1)[:, :h_old + h_pad, :]
        inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :w_old + w_pad]

        coord = make_coord((round(scale * (h_old + h_pad)), round(scale * (w_old + w_pad))), flatten=False).unsqueeze(
            0).cuda()
        cell = torch.ones_like(cell)
        cell[:, 0] *= 2 / inp.shape[-2] / scale
        cell[:, 1] *= 2 / inp.shape[-1] / scale
    else:
        h_pad = 0
        w_pad = 0
        coord = coord
        cell = cell

    cell_factor = max(scale / scale_max, 1)
    pred = batched_predict_fast(model, inp.unsqueeze(0).cuda(), coord, cell_factor * cell.cuda(), style_image,bsize=300)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1)

    if args.window_size != 0:
        shape = [3, round(scale * (h_old + h_pad)), round(scale * (w_old + w_pad))]
    else:
        shape = [3, h, w]
    pred = pred.view(*shape).contiguous()
    pred = pred[..., :h, :w]

    pred = pred.cpu()

    transforms.ToPILImage()(pred).save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='./load/LR', help='Path to input folder')
    parser.add_argument('--output_folder', default='./visual/hiifvs')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--scale', type=float, required=True, help='Scaling factor')
    parser.add_argument('--gpu', default='1', help='GPU id to use')
    parser.add_argument('--window_size', type=int, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    state_dict = torch.load(args.model)['model']
    new_sd = OrderedDict()
    for key, value in state_dict['sd'].items():
        new_key = key.replace('module.', '')
        new_sd[new_key] = value
    state_dict['sd'] = new_sd
    model = models.make(state_dict, load_sd=True).cuda()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in os.listdir(args.input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff','.tif')):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, filename)
            latent_index = random.randint(0, 49999)
            sd_features_cache = torch.load("./results/sd_features_cache.pt")
            sd_features = sd_features_cache[latent_index]['latent'].to("cuda")

            process_image(input_path, output_path, model, sd_features, args.scale)
