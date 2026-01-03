import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

import torch.fft


def compute_hi_coord(coord, n):
    coord_clip = torch.clip(coord - 1e-9, 0., 1.)
    coord_bin = ((coord_clip * 2 ** (n + 1)).floor() % 2)
    return coord_bin

class StyleAdapterAgnostic(nn.Module):
    def __init__(self, c_out, mid=588):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, mid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mid, 2 * c_out)

        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        """
        z: [B, 4, H, W] (例如 Stable Diffusion latent)
        """
        h = self.pre(z)                          # [B, mid, H, W]
        h = self.pool(h).squeeze(-1).squeeze(-1) # [B, mid]

        gb = self.fc(h)                          # [B, 2 * c_out]
        gamma, beta = gb.chunk(2, dim=-1)        #  [B, c_out]

        alpha = torch.sigmoid(self.gate)

        gamma = gamma * alpha
        beta = beta * alpha

        return gamma, beta


class SFTBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.gamma_scale = nn.Parameter(torch.zeros(1, C))
        self.beta_scale = nn.Parameter(torch.zeros(1, C))

        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, gamma, beta):
        """
        x:     [B, C, H, W] 或 [B, N, C]
        gamma: [B, C]
        beta:  [B, C]
        """
        is_4d = (x.dim() == 4)

        if is_4d:
            # [B, C] -> [B, C, 1, 1]
            if gamma.dim() == 2:
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            if beta.dim() == 2:
                beta = beta.unsqueeze(-1).unsqueeze(-1)

            gamma_scale = self.gamma_scale.unsqueeze(-1).unsqueeze(-1)
            beta_scale = self.beta_scale.unsqueeze(-1).unsqueeze(-1)
        else:
            # [B, C] -> [B, 1, C]
            if gamma.dim() == 2:
                gamma = gamma.unsqueeze(1)
            if beta.dim() == 2:
                beta = beta.unsqueeze(1)

            gamma_scale = self.gamma_scale
            beta_scale = self.beta_scale

        residual = x * (1.0 + gamma * gamma_scale) + beta * beta_scale

        alpha = torch.sigmoid(self.gate)
        out = (1.0 - alpha) * x + alpha * residual

        return out


class ContextAwareFeatureModulation(nn.Module):
    def __init__(self, regression_channels=32, main_channels=256, reduction=8):
        super(ContextAwareFeatureModulation, self).__init__()

        self.regression_channels = regression_channels
        self.main_channels = main_channels

        self.feature_align = nn.Sequential(
            nn.Conv2d(regression_channels, main_channels // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(main_channels // 4, main_channels // 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.spatial_context = nn.Sequential(
            nn.Conv2d(main_channels // 2, main_channels // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(main_channels // 4, 1, 1, 1, 0),
            nn.Sigmoid()
        )

        self.channel_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(main_channels // 2, main_channels // reduction, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(main_channels // reduction, main_channels, 1, 1, 0),
            nn.Sigmoid()
        )

        self.cross_scale_fusion = nn.Sequential(
            nn.Conv2d(main_channels + main_channels // 2, main_channels, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(main_channels, main_channels, 3, 1, 1),
        )

        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, regression_feat, main_feat):
        """
        Args:
            regression_feat: [bs, 32, 128, 128]
            main_feat: [bs, 256, h, w]
        Returns:
            modulated_feat: [bs, 256, h, w]
        """
        bs, main_c, h, w = main_feat.shape

        if regression_feat.size(-1) != h or regression_feat.size(-2) != w:
            regression_feat_resized = F.interpolate(
                regression_feat, size=(h, w), mode='bilinear', align_corners=False
            )
        else:
            regression_feat_resized = regression_feat

        aligned_feat = self.feature_align(regression_feat_resized)

        spatial_attention = self.spatial_context(aligned_feat)

        channel_attention = self.channel_context(aligned_feat)

        spatially_modulated = main_feat * spatial_attention

        channel_modulated = spatially_modulated * channel_attention

        fused_input = torch.cat([channel_modulated, aligned_feat], dim=1)

        fused_feat = self.cross_scale_fusion(fused_input)

        modulated_feat = main_feat + self.residual_weight * fused_feat

        return modulated_feat



@register('hiif')
class hiif(nn.Module):

    def __init__(self, encoder_spec, blocks=16, hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)

        self.n_hi_layers = 6
        self.fc_layers = nn.ModuleList(
            [
                MLP_with_shortcut(hidden_dim * 4 + 2 + 2 if d == 0 else hidden_dim + 2,
                                  3 if d == self.n_hi_layers - 1 else hidden_dim,256) \
                for d in range(self.n_hi_layers)
            ]
        )

        self.CAFM = ContextAwareFeatureModulation(
            regression_channels=32,
            main_channels=256
        )


        self.conv0 = qkv_attn(hidden_dim, blocks)
        self.conv1 = qkv_attn(hidden_dim, blocks)
        self.adapter = StyleAdapterAgnostic(c_out=hidden_dim)
        self.SFTBlock = SFTBlock(C=hidden_dim)

    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        self.feat = self.freq(self.feat)
        return self.feat

    def query_rgb(self, coord, cell,style_image,regression_feature):
        feat = (self.feat)
        grid = 0
        B, C, H, W = feat.shape
        feat = self.CAFM(regression_feature,feat)
        #print(f"this is feat shape:{feat.shape}") [bs,256,48,48]
        #print(f"this is regression:{regression_feature.shape}") [bs,256,1]
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                if style_image.dim() == 3:
                    style_image = style_image.unsqueeze(0)
                #print(f"this is style_image:{style_image.shape}")#[bs,4,64,64]
                gamma,beta = self.adapter(style_image.float())
                feat_ = self.SFTBlock(feat_,gamma,beta)



                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2] / 2
                rel_coord[:, 1, :, :] *= feat.shape[-1] / 2
                rel_coord_n = rel_coord.permute(0, 2, 3, 1).reshape(rel_coord.shape[0], -1, rel_coord.shape[1])

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                preds.append(feat_)
                if vx == -1 and vy == -1:
                    # Local coord
                    rel_coord_mask = (rel_coord_n > 0).float()
                    rxry = torch.tensor([rx, ry], device=coord.device)[None, None, :]
                    local_coord = rel_coord_mask * rel_coord_n + (1. - rel_coord_mask) * (rxry - rel_coord_n)

        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t

        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(1)

        grid = torch.cat([*preds, rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])], dim=1)

        B, C_g, H, W = grid.shape
        grid = grid.permute(0, 2, 3, 1).reshape(B, H * W, C_g)

        for n in range(self.n_hi_layers):
            hi_coord = compute_hi_coord(local_coord, n)
            if n == 0:
                x = torch.cat([grid] + [hi_coord], dim=-1)
            else:
                x = torch.cat([x] + [hi_coord], dim=-1)
            x = self.fc_layers[n](x)
            if n == 0:
                x = self.conv0(x)
                x = self.conv1(x)

        result = x.permute(0, 2, 1).reshape(B, 3, H, W)

        ret = result + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear', \
                                  padding_mode='border', align_corners=False)
        return ret

    def forward(self, inp, coord, cell,style_image,regression_features):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell,style_image,regression_features)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLP_with_shortcut(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if x.shape[-1] == short_cut.shape[-1]:
            x = x + short_cut
        return x

class qkv_attn(nn.Module):
    def __init__(self, midc, heads):
        super().__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Linear(midc, midc * 3, bias=True)

        self.kln = nn.LayerNorm(self.headc)
        self.vln = nn.LayerNorm(self.headc)
        self.sm = nn.Softmax(dim=-1)

        self.proj1 = nn.Linear(midc, midc)
        self.proj2 = nn.Linear(midc, midc)

        self.proj_drop = nn.Dropout(0.)

        self.act = nn.GELU()

    def forward(self, x):
        B, HW, C = x.shape
        bias = x

        qkv = self.qkv_proj(x).reshape(B, HW, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1) # B, heads, HW, headc

        k = self.kln(k)
        v = self.vln(v)

        v = torch.matmul(k.transpose(-2, -1), v) / (HW)
        # v = self.sm(v)
        v = torch.matmul(q, v)
        v = v.permute(0, 2, 1, 3).reshape(B, HW, C)

        ret = v + bias
        bias = self.proj2(self.act(self.proj1(ret))) + bias

        return bias
