import torch
import torch.nn as nn
import torch.optim as optim
from models import networks
from .base_model import BaseModel
from models.resolution_regression_net import ResolutionRegressionNet
import numpy as np


class ResolutionRegressionNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--base_resolution', type=float, default=0.1,
                            help='base resolution value in m/px')
        parser.add_argument('--upsample_range_min', type=float, default=2.0,
                            help='minimum upsampling factor')
        parser.add_argument('--upsample_range_max', type=float, default=10.0,
                            help='maximum upsampling factor')
        parser.add_argument('--downsample_range_min', type=float, default=2.0,
                            help='minimum downsampling factor')
        parser.add_argument('--downsample_range_max', type=float, default=10.0,
                            help='maximum downsampling factor')

        parser.add_argument('--lambda_mse', type=float, default=1.0,
                            help='weight for MSE loss')
        parser.add_argument('--lambda_mae', type=float, default=0.5,
                            help='weight for MAE loss')
        parser.add_argument('--lambda_smooth_l1', type=float, default=0.5,
                            help='weight for Smooth L1 loss')
        parser.add_argument('--use_log_scale', action='store_true',
                            help='whether to use log scale for resolution values')

        parser.add_argument('--regression_lr', type=float, default=0.0001,
                            help='initial learning rate for regression network')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            help='weight decay for optimizer')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netR = ResolutionRegressionNet()
        self.netR = networks.define_network(opt, self.netR)

        self.model_names = ['R']
        self.load_model_names = ['R']

        self.loss_names = ['Total', 'High_Weighted', 'Medium_Weighted', 'Low_Weighted']

        self.visual_names = ['concatenated_images', 'predicted_resolutions', 'target_resolutions']

        if self.isTrain:
            self.criterion_mse = nn.MSELoss()
            self.criterion_mae = nn.L1Loss()
            self.criterion_smooth_l1 = nn.SmoothL1Loss()

            self.optimizer_R = optim.Adam(
                self.netR.parameters(),
                lr=opt.regression_lr if hasattr(opt, 'regression_lr') else opt.lr,
                betas=(opt.beta1, 0.99),
                weight_decay=opt.weight_decay if hasattr(opt, 'weight_decay') else 1e-4
            )
            self.optimizers = [self.optimizer_R]

        self.base_resolution = opt.base_resolution if hasattr(opt, 'base_resolution') else 0.1
        self.use_log_scale = opt.use_log_scale if hasattr(opt, 'use_log_scale') else False

        self.lambda_mse = opt.lambda_mse if hasattr(opt, 'lambda_mse') else 1.0
        self.lambda_mae = opt.lambda_mae if hasattr(opt, 'lambda_mae') else 0.5
        self.lambda_smooth_l1 = opt.lambda_smooth_l1 if hasattr(opt, 'lambda_smooth_l1') else 0.5

        self.eval_metrics = {
            'mae': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'r2_score': 0.0
        }

    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters

        self.concatenated_images = input['images'].to(self.opt.data_device)

        self.target_resolutions = input['labels'].to(self.opt.data_device)

        if self.use_log_scale:
            self.target_resolutions = torch.log(self.target_resolutions + 1e-8)

        if 'img_paths' in input:
            self.image_paths = input['img_paths']

        if 'up_factor' in input:
            self.up_factors = input['up_factor']
        if 'down_factor' in input:
            self.down_factors = input['down_factor']

    def forward(self):
        self.predicted_resolutions = self.netR(self.concatenated_images)

        if self.use_log_scale:
            self.predicted_resolutions_original = torch.exp(self.predicted_resolutions) - 1e-8
        else:
            self.predicted_resolutions_original = self.predicted_resolutions

    def backward_R(self):
        self.loss_MSE = self.criterion_mse(self.predicted_resolutions, self.target_resolutions)
        self.loss_MAE = self.criterion_mae(self.predicted_resolutions, self.target_resolutions)
        self.loss_SmoothL1 = self.criterion_smooth_l1(self.predicted_resolutions, self.target_resolutions)

        self.loss_Total = (self.lambda_mse * self.loss_MSE +
                           self.lambda_mae * self.loss_MAE +
                           self.lambda_smooth_l1 * self.loss_SmoothL1)

        self.loss_Total.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_R.zero_grad()
        self.backward_R()
        self.optimizer_R.step()

    def compute_eval_metrics(self):
        if not hasattr(self, 'predicted_resolutions') or not hasattr(self, 'target_resolutions'):
            return

        with torch.no_grad():
            if self.use_log_scale:
                pred = self.predicted_resolutions_original.cpu().numpy()
                target = torch.exp(self.target_resolutions).cpu().numpy() - 1e-8
            else:
                pred = self.predicted_resolutions.cpu().numpy()
                target = self.target_resolutions.cpu().numpy()

            self.eval_metrics['mae'] = np.mean(np.abs(pred - target))

            self.eval_metrics['rmse'] = np.sqrt(np.mean((pred - target) ** 2))

            self.eval_metrics['mape'] = np.mean(np.abs((pred - target) / (target + 1e-8))) * 100

            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            self.eval_metrics['r2_score'] = 1 - (ss_res / (ss_tot + 1e-8))

    def get_current_losses(self):
        losses = {}
        for name in self.loss_names:
            if hasattr(self, f'loss_{name}'):
                losses[name] = getattr(self, f'loss_{name}').item()
        return losses

    def get_current_metrics(self):
        self.compute_eval_metrics()
        return self.eval_metrics.copy()

    def get_current_visuals(self):
        visuals = {}

        if hasattr(self, 'concatenated_images'):
            images = self.concatenated_images[0].cpu()  # [9, H, W]
            img1 = images[0:3]
            img2 = images[3:6]
            img3 = images[6:9]

            visuals['high_res_img'] = img1
            visuals['medium_res_img'] = img2
            visuals['low_res_img'] = img3

        if hasattr(self, 'predicted_resolutions_original'):
            visuals['predicted_resolutions'] = self.predicted_resolutions_original[0].cpu()  # [3]

        if hasattr(self, 'target_resolutions'):
            if self.use_log_scale:
                visuals['target_resolutions'] = (torch.exp(self.target_resolutions[0]) - 1e-8).cpu()
            else:
                visuals['target_resolutions'] = self.target_resolutions[0].cpu()

        return visuals

    def print_current_predictions(self, num_samples=1):
        if not hasattr(self, 'predicted_resolutions_original'):
            return

        print("\n=== Resolution Prediction Results ===")
        for i in range(min(num_samples, self.predicted_resolutions_original.size(0))):
            if self.use_log_scale:
                pred = self.predicted_resolutions_original[i].cpu().numpy()
                target = (torch.exp(self.target_resolutions[i]) - 1e-8).cpu().numpy()
            else:
                pred = self.predicted_resolutions[i].cpu().numpy()
                target = self.target_resolutions[i].cpu().numpy()

            print(f"Sample {i + 1}:")
            print(f"  Predicted: [{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}] m/px")
            print(f"  Target:    [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}] m/px")
            print(
                f"  Error:     [{abs(pred[0] - target[0]):.4f}, {abs(pred[1] - target[1]):.4f}, {abs(pred[2] - target[2]):.4f}] m/px")

            if hasattr(self, 'up_factors') and hasattr(self, 'down_factors'):
                print(f"  Up factor: {self.up_factors[i]:.2f}, Down factor: {self.down_factors[i]:.2f}")
        print("=" * 40)

    def load_pretrain_model(self):
        if hasattr(self.opt, 'pretrain_model_path') and self.opt.pretrain_model_path:
            print(f'Loading pretrained model from {self.opt.pretrain_model_path}')
            weight = torch.load(self.opt.pretrain_model_path, map_location=self.device)

            if hasattr(self.netR, 'module'):
                self.netR.module.load_state_dict(weight)
            else:
                self.netR.load_state_dict(weight)
            print('Pretrained model loaded successfully!')