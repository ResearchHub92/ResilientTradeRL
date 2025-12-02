from src.utils.config import YEARS, NUM_NODES, COUNTRIES, SECTOR_CODES
import pandas as pd
import ast
from src.utils.sr_calculator import compute_dynamic_target_sr
from src.utils.sr_metric import compute_sr_wr
from functools import lru_cache
from torch_scatter import scatter_add
import numpy as np
import random
import torch
import gymnasium as gym
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def find_actual_complement_pairs(edge_index, edge_weight, n=5):
    """Finding complementary segment pairs based on real data"""
    if edge_weight.numel() == 0:
        return []

    degrees = scatter_add(torch.ones_like(edge_weight),
                          edge_index[0], dim=0, dim_size=NUM_NODES)
    top_nodes = torch.topk(degrees, min(20, NUM_NODES)).indices.tolist()

    pairs = []
    for i in range(min(n, len(top_nodes) // 2)):
        if len(top_nodes) >= 2:
            src = random.choice(top_nodes)
            dst = random.choice([x for x in top_nodes if x != src])
            pairs.append((src, dst))

    return pairs


@lru_cache(maxsize=15)
def _load_year_cached(year):
    path = f"data/processed/WIOT_{year}_filtered_with_codes.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"file {path} Not found!")

    print(f" Loading year data {year}...")

    try:
        # load CSV
        df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        print(f" Year data dimensions {year}: {df.shape}")

        # Check that the data is not empty
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError(f"Year data{year} They are empty !")

        non_zero_count = (df != 0).sum().sum()
        print(f" Number of non-zero edges per year {year}: {non_zero_count}")
        if non_zero_count == 0:
            raise ValueError(
                f"No non-zero edges in the year data {year} Does not exist!")

        # Function to parse indexes and columns
        def parse_index(index):
            if isinstance(index, str):
                try:
                    parsed = ast.literal_eval(index)
                    if isinstance(parsed, tuple) and len(parsed) == 2:
                        return parsed[0], parsed[1]
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(index, tuple) and len(index) == 2:
                return index[0], index[1]
            raise ValueError(f"Index format/Unexpected column: {index}")

        print(f" Processing indexes for the year {year}...")
        src_codes = []
        src_countries = []
        for idx in df.index:
            sector_code, country_code = parse_index(idx)
            src_codes.append(sector_code)
            src_countries.append(country_code)

        dst_codes = []
        dst_countries = []
        for col in df.columns:
            sector_code, country_code = parse_index(col)
            dst_codes.append(sector_code)
            dst_countries.append(country_code)

        print(
            f" Sample src_codes: {src_codes[:5]}, src_countries: {src_countries[:5]}")
        print(
            f" Sample dst_codes: {dst_codes[:5]}, dst_countries: {dst_countries[:5]}")

        # Checking the validity of codes
        if not all(code in SECTOR_CODES for code in set(src_codes + dst_codes)):
            invalid_codes = set(src_codes + dst_codes) - set(SECTOR_CODES)
            raise ValueError(f"Invalid section codes found: {invalid_codes}")
        if not all(country in COUNTRIES for country in set(src_countries + dst_countries)):
            invalid_countries = set(
                src_countries + dst_countries) - set(COUNTRIES)
            raise ValueError(
                f"Invalid section codes found : {invalid_countries}")

        # Map of the country to offset
        node_map = {country: i * len(SECTOR_CODES)
                                     for i, country in enumerate(COUNTRIES)}

        # Finding indexes nonzero
        src, dst = np.nonzero(df.values)
        weights = df.values[src, dst]

        # Construction edge_index
        src_nodes = []
        for i in src:
            sector_code = src_codes[i]
            country = src_countries[i]
            sector_idx = SECTOR_CODES.index(sector_code)
            src_nodes.append(node_map[country] + sector_idx)

        dst_nodes = []
        for i in dst:
            sector_code = dst_codes[i]
            country = dst_countries[i]
            sector_idx = SECTOR_CODES.index(sector_code)
            dst_nodes.append(node_map[country] + sector_idx)

        edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float)

        class Graph:
            def __init__(self, edge_index, edge_weight):
                self.edge_index = edge_index
                self.edge_weight = edge_weight

        return Graph(edge_index, edge_weight)

    except Exception as e:
        print(f"Error loading year {year}: {str(e)}")
        raise


class WIODEnv_v2(gym.Env):
    def __init__(self, shock_mode=False, device="cpu", max_steps=20):
        super().__init__()
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0
        self.shock_mode = shock_mode
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(NUM_NODES, 2), dtype=np.float32)
        self.graph = None
        self.year = None
        self.initial_edge_count = 0
        self.prev_dist = float('inf')
        self.target_sr = None  # Value in reset

    def reset(self, seed=None, options=None):
        # torch.manual_seed(42)
        super().reset(seed=seed)
        self.year = random.choice(YEARS) if options is None or not isinstance(
            options, dict) else options.get("year", random.choice(YEARS))
        original = _load_year_cached(self.year)
        self.graph = type('Graph', (), {
            'edge_index': original.edge_index.clone().to(self.device),
            'edge_weight': original.edge_weight.clone().to(self.device)
        })()
        self.initial_edge_count = self.graph.edge_index.shape[1]
        self.current_step = 0

        # Calculate target_sr only in reset
        if self.target_sr is None:
            self.target_sr = compute_dynamic_target_sr(
                self.graph.edge_index, self.graph.edge_weight)
            print(f" target_sr Dynamics calculated: {self.target_sr:.4f}")

        if self.shock_mode:
            self.apply_shock()

        self.initial_sr = self._compute_sr()
        self.prev_dist = abs(self.initial_sr - self.target_sr)

        return self._get_obs(), {"sr": self.initial_sr}

    def _get_obs(self):
        if self.graph.edge_weight.numel() == 0:
            out_strength = torch.zeros(NUM_NODES, device=self.device)
            in_strength = torch.zeros(NUM_NODES, device=self.device)
        else:
            out_strength = scatter_add(
                self.graph.edge_weight, self.graph.edge_index[0], dim=0, dim_size=NUM_NODES)
            in_strength = scatter_add(
                self.graph.edge_weight, self.graph.edge_index[1], dim=0, dim_size=NUM_NODES)

        total = out_strength.sum() + 1e-12
        out_strength = out_strength / total
        in_strength = in_strength / total

        obs = torch.stack([out_strength, in_strength], dim=1)
        return obs.cpu().numpy()

    def _compute_sr(self):
        return compute_sr_wr(self.graph.edge_index, self.graph.edge_weight, NUM_NODES)

    # #**************************************
    # def _simulate_cascade(self, steps=5):
    #     if self.graph is None or self.graph.edge_index is None or self.graph.edge_weight is None:
    #         raise ValueError("Graph is not initialized. Ensure reset() has been called with valid data.")
    #     cascade_nodes = []
    #     graph_copy = (self.graph.edge_index.clone(), self.graph.edge_weight.clone())  # Copy to preserve the original graph

    #     for step in range(steps):
    #         degrees = scatter_add(torch.ones_like(graph_copy[1]), graph_copy[0][0], dim=0, dim_size=NUM_NODES)
    #         isolated = (degrees == 0).nonzero().squeeze().tolist()
    #         cascade_nodes.append(isolated)

    #         if isolated:  # If there is an isolated node
    #             mask = ~torch.isin(graph_copy[0][0], torch.tensor(isolated, device=self.device))
    #             graph_copy = (graph_copy[0][:, mask], graph_copy[1][mask])  # Graph update

    #     return cascade_nodes

    def apply_shock(self, shock_fraction=0.1):
        # torch.manual_seed(42)
        """Remove edges with high SR (highest impacts)""""
        print(f" Shock by removal {shock_fraction*100}% High-impact ridges...")
        edge_weights = self.graph.edge_weight.numpy()
        edge_index = self.graph.edge_index.numpy()
        degrees = np.bincount(edge_index[0], minlength=NUM_NODES)  # Node degree
        num_edges_to_remove = int(len(edge_weights) * shock_fraction)
        
        # Calculate the influence of edges (weight × sum of node degrees)
        impact = edge_weights * (degrees[edge_index[0]] + degrees[edge_index[1]])
        remove_indices = np.argsort(impact)[-num_edges_to_remove:]  # Highest Impacts
        
        # Edge removal
        mask = np.ones(len(edge_weights), dtype=bool)
        mask[remove_indices] = False
        self.graph.edge_index = torch.tensor(edge_index[:, mask], dtype=torch.long)
        self.graph.edge_weight = torch.tensor(edge_weights[mask], dtype=torch.float)
        
        #SR Update
        self.current_sr = self._compute_sr()
        print(f" Number of edges removed: {num_edges_to_remove}")
        return num_edges_to_remove  # Return the number for info



    def _apply_action(self, action):
        # Apply measures PPO
        if action == 0:  # no-op
            pass
        elif action == 1 and self.graph.edge_weight.numel() >= 5:  # remove 5 random low-weight edges
            if self.graph.edge_weight.numel() > 0:
                idx = torch.argsort(self.graph.edge_weight)[:5]  # low-weight
                keep_mask = torch.ones(self.graph.edge_weight.numel(), dtype=torch.bool, device=self.device)
                keep_mask[idx] = False
                self.graph.edge_index = self.graph.edge_index[:, keep_mask]
                self.graph.edge_weight = self.graph.edge_weight[keep_mask]
        elif action == 2:  # add 5 random edges
            new_edges = torch.randint(0, NUM_NODES, (2, 5), device=self.device)
            new_weights = torch.rand(5, device=self.device) * 10  # random weights
            self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)
            self.graph.edge_weight = torch.cat([self.graph.edge_weight, new_weights])
        elif action == 3:  # add complement pairs
            pairs = find_actual_complement_pairs(self.graph.edge_index, self.graph.edge_weight, n=10)
            if pairs:
                new_edges = torch.tensor(pairs).T.to(self.device)
                new_weights = torch.rand(len(pairs), device=self.device) * 10
                self.graph.edge_index = torch.cat([self.graph.edge_index, new_edges], dim=1)
                self.graph.edge_weight = torch.cat([self.graph.edge_weight, new_weights])
        elif action == 4 and self.graph.edge_weight.numel() >= 10:  # weaken top 10 edges
            idx = torch.topk(self.graph.edge_weight, 10).indices
            self.graph.edge_weight[idx] *= 0.5
        elif action == 5 and self.graph.edge_weight.numel() >= 10:  # strengthen top 10 edges
            idx = torch.topk(self.graph.edge_weight, 10).indices
            self.graph.edge_weight[idx] *= 1.5

    def compute_reward(self, prev_sr, current_sr):
        """Compensation calculation for maintaining SR around dynamic target"""
        target = self.target_sr  #Using dynamic targets
        prev_dist = abs(prev_sr - target)
        current_dist = abs(current_sr - target)
        reward = (prev_dist - current_dist) * 100 - current_dist * 10  # Encouragement to stay close
        return reward

    def step(self, action):
        self._apply_action(action)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            current_sr = self._compute_sr()
            current_dist = abs(current_sr - self.target_sr)
            prev_dist = self.prev_dist
            reward = self.compute_reward(self.initial_sr, current_sr)  # Use the new function
            self.prev_dist = current_dist
        else:
            current_sr = 0.0
            reward = 0.0
            current_dist = self.prev_dist
        terminated = self.current_step >= self.max_steps
        info = {"sr": current_sr} if terminated else {}
        return self._get_obs(), reward, terminated, False, info
    
    # def step(self, action):
    #     self._apply_action(action)
    #     self.current_step += 1
    #     num_edges_removed = 0
    #     if self.current_step == 1 and self.shock_mode:  # Shock is only applied on reset. 
    #         num_edges_removed = self.apply_shock()  # Recall or save to reset
    #     if self.current_step >= self.max_steps:
    #         current_sr = self._compute_sr()
    #         current_dist = abs(current_sr - self.target_sr)
    #         prev_dist = self.prev_dist
    #         reward = self.compute_reward(self.initial_sr, current_sr)
    #         self.prev_dist = current_dist
    #     else:
    #         current_sr = 0.0
    #         reward = 0.0
    #         current_dist = self.prev_dist
    #     terminated = self.current_step >= self.max_steps
    #     info = {"sr": current_sr, "edges_removed": num_edges_removed} if terminated else {"edges_removed": num_edges_removed}
    #     return self._get_obs(), reward, terminated, False, info

    def _get_degree_dist(self):
        out_strength = scatter_add(torch.ones_like(self.graph.edge_weight), self.graph.edge_index[0], dim=0, dim_size=NUM_NODES)
        return out_strength.tolist()

    
    def _compute_collapse_fraction(self):
        isolated = (scatter_add(torch.ones_like(self.graph.edge_weight), self.graph.edge_index[0], dim=0, dim_size=NUM_NODES) == 0).sum().float() / NUM_NODES
        return isolated.item()  # Convert to scalar

    def _compute_flow_loss(self):
        initial_flow = self.initial_edge_count * self.graph.edge_weight.mean()
        current_flow = self.graph.edge_index.shape[1] * self.graph.edge_weight.mean()
        loss = max(0, (initial_flow - current_flow) / initial_flow)
        return loss.item()

    

    def _get_flow_centrality(self):
        flow = scatter_add(self.graph.edge_weight, self.graph.edge_index[1], dim=0, dim_size=NUM_NODES)
        return (flow / flow.sum()).tolist()