from src.utils.sr_metric import compute_pearson_assortativity
from torch_scatter import scatter_add
from stable_baselines3 import PPO
from src.envs.wiod_env import WIODEnv_v2  # Update path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


YEARS = list(range(2000, 2015))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/ppo_wiod_model"  # New model
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("Loading model PPO...")
model = PPO.load(MODEL_PATH, device=DEVICE)

results = []
print("Starting a 15-year evaluation on filtered data...")
i = 0
j = 0
for year in YEARS:
    print(f"  → year {year}...")
    env_normal = WIODEnv_v2(shock_mode=False, device=DEVICE, max_steps=20)
    _, _ = env_normal.reset(options={"year": year})
    # print(f"Graph after reset for year {year}: {env_normal.graph.edge_index.shape if env_normal.graph and env_normal.graph.edge_index is not None else 'None'}")
    before_sr = float(env_normal._compute_sr())
    before_pearson = compute_pearson_assortativity(
        env_normal.graph.edge_index, env_normal.graph.edge_weight, 224)
    env_shock = WIODEnv_v2(shock_mode=True, device=DEVICE, max_steps=20)
    # cascade_result = env_shock._simulate_cascade(steps=5)
    # results[-1]['Cascade_Nodes'] = str(cascade_result)
    obs, _ = env_shock.reset(options={"year": year})
    shock_sr = float(env_shock._compute_sr())
    if i == 0:

        in_degree = env_normal._get_degree_dist()
        out_degree = env_normal._get_degree_dist()
        i += 1
    else:
        out_degree = str(scatter_add(torch.ones_like(env_shock.graph.edge_weight),
                         env_shock.graph.edge_index[0], dim=0, dim_size=224).tolist())
        in_degree = str(scatter_add(torch.ones_like(env_shock.graph.edge_weight),
                        env_shock.graph.edge_index[1], dim=0, dim_size=224).tolist())
        # results.append({'In_Degree' : str(in_degree)  ,'Out_Degree' : str(out_degree)})
        # results[-1]['In_Degree'] = str(in_degree)  #  Addition dict
        # results[-1]['Out_Degree'] = str(out_degree)

    shock_pearson = compute_pearson_assortativity(
        env_shock.graph.edge_index, env_shock.graph.edge_weight, 224)
    baseline_sr = env_shock._compute_sr()  # without PPO

    terminated = False
    steps = 0
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env_shock.step(action)
        steps += 1
    after_pearson = compute_pearson_assortativity(
        env_shock.graph.edge_index, env_shock.graph.edge_weight, 224)
    after_sr = float(env_shock._compute_sr())
    delta_shock = shock_sr - before_sr
    delta_recovery = after_sr - shock_sr
    total_recovery = after_sr - before_sr
    degree_dist = env_shock._get_degree_dist()
    flow_cent = env_shock._get_flow_centrality()
    collapse_fraction = env_shock._compute_collapse_fraction()
    flow_loss = env_shock._compute_flow_loss()
    recovery_steps = steps
    latency = steps * env_shock._get_obs().shape[0]  # Estimate latency

    results.append({
        "Year": year,
        "Before_SR": round(before_sr, 4),
        "Shock_SR": round(shock_sr, 4),
        "After_SR": round(after_sr, 4),
        "ΔSR_Shock": round(delta_shock, 4),
        "ΔSR_Recovery": round(delta_recovery, 4),
        "Total_Recovery": round(total_recovery, 4),
        "Collapse_Fraction": round(collapse_fraction, 4),
        "Flow_Loss": round(flow_loss, 4),
        "Recovery_Steps": recovery_steps,
        "Latency": round(latency, 4),
        "Degree_Dist": degree_dist,
        "Flow_Cent": flow_cent,
        'In_Degree': in_degree,
        'Out_Degree': out_degree,
        'Before_Pearson': before_pearson,
        'Shock_Pearson': shock_pearson,
        'After_Pearson':  after_pearson,
        'Baseline_SR': baseline_sr,
    })

df = pd.DataFrame(results)
csv_path = RESULTS_DIR / "shock_results_filtered.csv"
df.to_csv(csv_path, index=False)
print(f"  Results saved: {csv_path}")

# SR plot of changes
plt.figure(figsize=(12, 7))
plt.plot(df["Year"], df["Before_SR"], 'o-', label="Before Shock", color="blue")
plt.plot(df["Year"], df["Shock_SR"], 's-', label="After Shock", color="red")
plt.plot(df["Year"], df["After_SR"], '^-', label="After PPO", color="green")
plt.axhline(y=0.5, color='black', linestyle='--', label="Target SR = 0.5")
plt.fill_between(df["Year"], 0.4, 0.6, color='lightgreen',
                 alpha=0.3, label="Natural Range [0.4, 0.6]")
plt.title("US-China Trade Shock Simulation & PPO Recovery (Filtered WIOD 2000–2014)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Assortativity Rank (SR)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plot_path = RESULTS_DIR / "sr_shock_filtered.png"
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"  Shape saved.: {plot_path}")

mean_before = df["Before_SR"].mean()
mean_shock = df["Shock_SR"].mean()
mean_after = df["After_SR"].mean()
success_rate = (df["After_SR"].between(0.5, 0.6)).mean() * 100

print("\n" + "="*55)
print("Summary of results  (Filtered data):")
print(f"Average SR before shock: {mean_before:.4f}")
print(f"Average SR after shock: {mean_shock:.4f}")
print(f"Average SR after PPO: {mean_after:.4f}")
print(f"Success rate  (In the interval  [0.6, 0.45]): {success_rate:.1f}%")
print("="*55)
