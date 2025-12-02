# import torch
# from src.envs.wiod_env import WIODEnv_v2
# from src.utils.sr_metric import compute_sr_wr

# # Code for calculation
# env = WIODEnv_v2(shock_mode=False)
# stability = []
# for year in range(2000, 2015):
#     env.reset(options={"year": year})
#     # Sample subgraph (e.g., nodes 0-55 for f"{count}")
#     sub_mask = (env.graph.edge_index[0] < 56) & (env.graph.edge_index[1] < 56)
#     sub_edge_index = env.graph.edge_index[:, sub_mask]
#     sub_edge_weight = env.graph.edge_weight[sub_mask]
#     sub_sr = compute_sr_wr(sub_edge_index, sub_edge_weight, 56)
#     stability.append(sub_sr)
# #print(stability)  # Save to new CSV: subgraph_stability.csv
# # Explanation: Run and add the stability list to CSV (Column Year, Subgraph_SR).

# COUNTRY_NODES = {
#     'f"{count}"': (0, 55),      # China
#     'USA': (56, 111),    # USA
#     'TWN': (112, 167),   # Taiwan
#     'MEX': (168, 223)    # Mexico
# }


import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.envs.wiod_env import WIODEnv_v2
from src.utils.sr_metric import compute_sr_wr

countrys = ['CHN', 'USA', 'TWN', 'MEX']


def compute_subgraph_stability(count):
    """Calculate the stability of the folded subgraph and save it to a CSV file"""

   # print(" Calculating the stability of the Chinese subgraph...")
    if count == 'CHN':
        start_node = 0
        end_node = 55
    elif count == 'USA':
        start_node = 56
        end_node = 111
    elif count == 'TWN':
        start_node = 112
        end_node = 167
    elif count == 'MEX':
        start_node = 168
        end_node = 222

    env = WIODEnv_v2(shock_mode=False)
    stability_data = []

    for year in range(2000, 2015):
        env.reset(options={"year": year})

        # Create the China subgraph (nodes 0-55)
        sub_mask = (env.graph.edge_index[0] >= start_node) & (env.graph.edge_index[0] <= end_node) & \
            (env.graph.edge_index[1] >= start_node) & (
                env.graph.edge_index[1] <= end_node)
        sub_edge_index = env.graph.edge_index[:, sub_mask]
        sub_edge_weight = env.graph.edge_weight[sub_mask]

        # Calculate SR for the Chinese subgraph
        sub_sr = compute_sr_wr(sub_edge_index, sub_edge_weight, end_node)

        stability_data.append({
            'Year': year,
            f'Subgraph_SR_{count}': sub_sr,
            'Nodes_Count': end_node,
            'Edges_Count': sub_mask.sum().item()
        })

        # print(f" year {year}: SR = {sub_sr:.4f}")

    #  Save to CSV file
    df_subgraph = pd.DataFrame(stability_data)
    df_subgraph.to_csv(f'subgraph_stability_{count}.csv', index=False)
    # print(" Data in 'subgraph_stability_{count}.csv' Saved ")

    return df_subgraph


def plot_subgraph_stability(count):
    """Drawing the stability diagram of the Chinese subgraph"""

    # Read stored data
    df_subgraph = pd.read_csv(f'subgraph_stability_{count}.csv')
    df_main = pd.read_csv('results/shock_results_filtered.csv')

    # Data integration
    df_combined = pd.merge(df_main, df_subgraph, on='Year', how='left')

    # Create a chart
    plt.figure(figsize=(14, 8))

    # Main SR graph and subgraph
    plt.subplot(2, 1, 1)
    plt.plot(df_combined['Year'], df_combined['Before_SR'], 'b-', marker='o',
             linewidth=2, label='All Network (Before_SR)', alpha=0.8)
    plt.plot(df_combined['Year'], df_combined[f'Subgraph_SR_{count}'], 'r-', marker='s',
             linewidth=2, label=f'{count} subgraph (Subgraph_SR)', alpha=0.8)

    plt.ylabel('Stability ratio (SR)')
    plt.title(
        f'Comparison of the stability of the entire network and the subgraph of {count}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(df_combined['Year'])

    # Difference chart
    plt.subplot(2, 1, 2)
    difference = df_combined['Before_SR'] - df_combined[f'Subgraph_SR_{count}']
    plt.bar(df_combined['Year'], difference, alpha=0.7, color='purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Difference (whole network - subgraph)')
    plt.xlabel('Year')
    plt.title(
        f'Stability difference between the entire network and the {count} subgraph')
    plt.grid(True, alpha=0.3)
    plt.xticks(df_combined['Year'])

    plt.tight_layout()
    plt.savefig(
        f'subgraph_vs_network_stability_{count}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df_combined


def comprehensive_analysis(count):
    """Comprehensive analysis with new data"""

    df_combined = pd.read_csv(f'subgraph_stability_{count}.csv')
    df_main = pd.read_csv('results/shock_results_filtered.csv')
    df_combined = pd.merge(df_main, df_combined, on='Year', how='left')

    # Calculate new indicators
    df_combined['Stability_Gap'] = df_combined['Before_SR'] - \
        df_combined[f'Subgraph_SR_{count}']
    df_combined['Relative_Stability'] = df_combined[f'Subgraph_SR_{count}'] / \
        df_combined['Before_SR']

    # print("\n Comprehensive analysis of China's subgraph stability:")
    # print("=" * 50)
    # print(f"Average stability of the entire network: {df_combined['Before_SR'].mean():.4f}")
    # print(f"Average stability of China subgraph: {df_combined['Subgraph_SR_{count}'].mean():.4f}")
    # print(f"Average stability gap: {df_combined['Stability_Gap'].mean():.4f}")
    # print(f"Average relative stability: {df_combined['Relative_Stability'].mean():.2%}")

    # Save comprehensive analysis
    df_combined.to_csv(f'comprehensive_analysis_{count}.csv', index=False)
    # print(" Comprehensive analysis in 'comprehensive_analysis.csv' Saved")

    return df_combined


# Main performance
if __name__ == "__main__":
    for count in countrys:
        # Step 1: Calculate and store subgraph data
        df_subgraph = compute_subgraph_stability(count)

        # Step 2: Draw the diagrams
        df_combined = plot_subgraph_stability(count)

        # Step 3: Comprehensive Analysis
        df_analysis = comprehensive_analysis(count)

        # print("\n All steps were completed successfully!")
        # print("Generated files:")
        # print("1. subgraph_stability_{count}.csv - Raw data under the graph")
        # print("2. subgraph_vs_network_stability.png - Comparison chart")
        # print("3. comprehensive_analysis.csv - Comprehensive analysis")
