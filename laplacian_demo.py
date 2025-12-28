import argparse
import os
from PIL import Image
import numpy as np
import cv2

import torch
from torchvision import transforms
import random
import models
from utils import make_coord
from test import batched_predict_fast
from collections import OrderedDict


def laplacian_variance_crop(image, crop_size=128, overlap_ratio=0.5):
    """
    Args:
        image: PIL Image
        crop_size: 128x128
        overlap_ratio: 0.5

    Returns:
        best_crop: PIL Image对象，最佳的crop
        best_position: tuple，最佳crop的位置(x, y)
        best_variance: float，最佳方差值
    """
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    h, w = gray.shape
    step = int(crop_size * (1 - overlap_ratio))  # 步长

    best_variance = -1
    best_position = (0, 0)

    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):

            patch = gray[y:y + crop_size, x:x + crop_size]

            laplacian = cv2.Laplacian(patch, cv2.CV_64F)


            variance = np.var(laplacian)


            if variance > best_variance:
                best_variance = variance
                best_position = (x, y)


    x, y = best_position
    best_crop = image.crop((x, y, x + crop_size, y + crop_size))

    return best_crop, best_position, best_variance


def bicubic_resample(image, scale):
    return image.resize((int(image.width * scale), int(image.height * scale)), Image.BICUBIC)


def process_image(input_path, output_path, model, style_image, scale, scale_max=4, laplacian_save_dir=None):
    img = Image.open(input_path).convert('RGB')

    if laplacian_save_dir is not None:
        crop_size = 128
        if img.width < crop_size or img.height < crop_size:
            new_size = crop_size * 2
            img_for_crop = img.resize((new_size, new_size), Image.BICUBIC)
            print(f"Image resized to {new_size}x{new_size} for cropping")
        else:
            img_for_crop = img


        best_crop, position, variance = laplacian_variance_crop(img_for_crop, crop_size)

        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        laplacian_output_path = os.path.join(laplacian_save_dir, f"{name}_laplacian_crop{ext}")
        best_crop.save(laplacian_output_path)
        print(f"Laplacian best crop saved: {laplacian_output_path}")
        print(f"Position: {position}, Variance: {variance:.4f}")

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
    pred = \
    batched_predict_fast(model, inp.unsqueeze(0).cuda(), coord, cell_factor * cell.cuda(), style_image, bsize=300)[0]
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
    parser.add_argument('--gpu', default='0', help='GPU id to use')
    parser.add_argument('--window_size', type=int, default='0')
    parser.add_argument('--save_laplacian_crops', default='True',action='store_true', help='Save Laplacian selected crops')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    state_dict = torch.load(args.model)['model']
    new_sd = OrderedDict()
    for key, value in state_dict['sd'].items():
        new_key = key.replace('module.', '')
        new_sd[new_key] = value
    state_dict['sd'] = new_sd
    model = models.make(state_dict, load_sd=True).cuda()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # 创建拉普拉斯crop保存目录
    # laplacian_save_dir = None
    # if args.save_laplacian_crops:
    #     laplacian_save_dir = '/data2/lrf/HIIF/results/test'
    #     if not os.path.exists(laplacian_save_dir):
    #         os.makedirs(laplacian_save_dir)
    #     print(f"Laplacian crops will be saved to: {laplacian_save_dir}")

    for filename in os.listdir(args.input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, filename)
            sd_features_cache = torch.load("./results/sd_features_cache.pt")
            latent_index = random.randint(0, 49999)
            sd_features = sd_features_cache[latent_index]['latent'].cuda()

            print(f"\nProcessing: {filename}")
            try:
                process_image(input_path, output_path, model, sd_features, args.scale,
                              laplacian_save_dir=laplacian_save_dir)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")