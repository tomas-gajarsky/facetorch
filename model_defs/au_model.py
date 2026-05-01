"""Native PyTorch implementation of the OpenGraphAU facial action unit model.

Uses timm's Swin Transformer backbone (swin_base_patch4_window7_224) instead of
the custom Microsoft implementation. The timm version is compatible with
torch.export for portable .pt2 serialization with dynamic batch sizes.

Architecture based on:
    - ME-GraphAU (CVI-SZU/ME-GraphAU) - IJCAI-ECAI 2022
    - OpenGraphAU (lingjivoo/OpenGraphAU)
"""

import math
import re

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# GNN + Head
# =============================================================================


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1).contiguous()
        x = self.relu(self.bn(x)).permute(0, 2, 1).contiguous()
        return x


def _normalize_digraph(A: torch.Tensor) -> torch.Tensor:
    # Build per-sample diagonal degree matrix directly from A to avoid
    # exporting a hard-coded device for identity tensor creation.
    node_degrees = A.detach().sum(dim=-1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.diag_embed(degs_inv_sqrt).to(dtype=A.dtype)
    return torch.bmm(torch.bmm(norm_degs_matrix, A), norm_degs_matrix)


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.neighbor_num = neighbor_num
        self.relu = nn.ReLU()
        self.U = nn.Linear(in_channels, in_channels)
        self.V = nn.Linear(in_channels, in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)
        self.U.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape
        si = x.detach()
        si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
        threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].reshape(b, n, 1)
        adj = (si >= threshold).float()
        A = _normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        return self.relu(x + self.bnv(aggregate + self.U(x)))


class AUHead(nn.Module):
    def __init__(self, in_channels, num_main_classes=27, num_sub_classes=14):
        super().__init__()
        self.in_channels = in_channels
        self.num_main_classes = num_main_classes
        self.num_sub_classes = num_sub_classes
        layers = [LinearBlock(in_channels, in_channels) for _ in range(num_main_classes)]
        self.main_class_linears = nn.ModuleList(layers)
        self.gnn = GNN(in_channels, num_main_classes)
        self.main_sc = nn.Parameter(torch.zeros(num_main_classes, in_channels))
        self.sub_sc = nn.Parameter(torch.zeros(num_sub_classes, in_channels))
        self.sub_list = [0, 1, 2, 4, 7, 8, 11]
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.main_sc)
        nn.init.xavier_uniform_(self.sub_sc)

    def forward(self, x):
        f_u = [layer(x).unsqueeze(1) for layer in self.main_class_linears]
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)
        f_v = self.gnn(f_v)
        b, n, c = f_v.shape
        main_sc = F.normalize(self.relu(self.main_sc), p=2, dim=-1)
        main_cl = (F.normalize(f_v, p=2, dim=-1) * main_sc.reshape(1, n, c)).sum(dim=-1)
        sub_cl = []
        for i, index in enumerate(self.sub_list):
            main_au = F.normalize(f_v[:, index], p=2, dim=-1)
            sc_l = F.normalize(self.relu(self.sub_sc[2 * i]), p=2, dim=-1)
            sc_r = F.normalize(self.relu(self.sub_sc[2 * i + 1]), p=2, dim=-1)
            sub_cl.append((main_au * sc_l.reshape(1, c)).sum(dim=-1, keepdim=True))
            sub_cl.append((main_au * sc_r.reshape(1, c)).sum(dim=-1, keepdim=True))
        return torch.cat([main_cl, torch.cat(sub_cl, dim=-1)], dim=-1)


# =============================================================================
# Full Model
# =============================================================================


class OpenGraphAU(nn.Module):
    """OpenGraphAU model for facial action unit detection.

    Architecture: timm Swin Transformer Base backbone + GNN head with 27 main
    and 14 sub action unit classes (41 total).
    """

    def __init__(self, num_main_classes: int = 27, num_sub_classes: int = 14):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            num_classes=0,
            global_pool="",
            pretrained=False,
        )
        in_channels = self.backbone.num_features
        out_channels = in_channels // 2
        self.global_linear = LinearBlock(in_channels, out_channels)
        self.head = AUHead(out_channels, num_main_classes, num_sub_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.contiguous().reshape(x.shape[0], -1, x.shape[-1])
        x = self.global_linear(x)
        return self.head(x)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Remap downsample keys: custom layers.N.downsample -> timm layers.N+1.downsample
        mapped = {}
        ds_pattern = re.compile(r"^backbone\.layers\.(\d+)\.downsample\.")
        for k, v in state_dict.items():
            m = ds_pattern.match(k)
            if m:
                old_idx = int(m.group(1))
                new_key = k.replace(
                    f"backbone.layers.{old_idx}.downsample.",
                    f"backbone.layers.{old_idx + 1}.downsample.",
                )
                mapped[new_key] = v
            else:
                mapped[k] = v
        # timm computes relative_position_index and attn_mask buffers on init,
        # so missing keys for those are expected
        return super().load_state_dict(mapped, strict=False, assign=assign)
