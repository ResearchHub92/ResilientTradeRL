# src/utils/sr_metric.py
import torch
from torch_scatter import scatter_add
import numpy as np


def compute_sr_wr(edge_index, edge_weight, num_nodes):
    idx_src, w = edge_index[0], edge_weight
    out_strength = scatter_add(w, idx_src, dim=0, dim_size=num_nodes)
    idx_dst, w = edge_index[1], edge_weight
    in_strength = scatter_add(w, idx_dst, dim=0, dim_size=num_nodes)
    rank_out = torch.argsort(torch.argsort(out_strength)).float()
    rank_in = torch.argsort(torch.argsort(in_strength)).float()
    W = edge_weight.sum().float()
    r_src = rank_out[edge_index[0]]
    r_tar = rank_in[edge_index[1]]
    r_src_bar = (edge_weight * r_src).sum() / W
    r_tar_bar = (edge_weight * r_tar).sum() / W
    numer = (edge_weight * (r_src - r_src_bar) * (r_tar - r_tar_bar)).sum()
    denom_src = (edge_weight * (r_src - r_src_bar) ** 2).sum()
    denom_tar = (edge_weight * (r_tar - r_tar_bar) ** 2).sum()
    sr = numer / (torch.sqrt(denom_src * denom_tar) + 1e-12)
    return float(sr)


def compute_pearson_assortativity(edge_index, edge_weight, num_nodes):
    idx_src, w = edge_index[0], edge_weight
    out_strength = scatter_add(w, idx_src, dim=0, dim_size=num_nodes)
    idx_dst, w = edge_index[1], edge_weight
    in_strength = scatter_add(w, idx_dst, dim=0, dim_size=num_nodes)
    mean_out = out_strength.mean()
    mean_in = in_strength.mean()
    cov = torch.sum((out_strength - mean_out) *
                    (in_strength - mean_in)) / num_nodes
    std_out = torch.sqrt(torch.sum((out_strength - mean_out) ** 2) / num_nodes)
    std_in = torch.sqrt(torch.sum((in_strength - mean_in) ** 2) / num_nodes)
    pearson = cov / (std_out * std_in + 1e-12)
    return float(pearson)

# def compute_dynamic_target_sr():
#     """Calculate the dynamic average target_sr based on the pre-shock SR for all years"""
#     env = WIODEnv_v2(shock_mode=False)
#     sr_values = []

#     for year in YEARS:
#         obs, info = env.reset(options={"year": year})
#         initial_sr = info["sr"]  # take SR از info
#         sr_values.append(initial_sr.item() if torch.is_tensor(initial_sr) else initial_sr)

#     mean_sr = np.mean(sr_values)
#     print(f" Average pre-shock SR for the years {YEARS[0]}-{YEARS[-1]}: {mean_sr:.4f}")
#     return mean_sr
