# import networkx as nx
# import matplotlib.pyplot as plt
# import ast
# import pandas as pd

# df = pd.read_csv('results/shock_results_filtered.csv')
# flow_cent = ast.literal_eval(df['Flow_Cent'][0])

# G = nx.complete_graph(224)
# nx.set_node_attributes(G, {i: flow_cent[i] for i in range(224)}, 'centrality')

# pos = nx.spring_layout(G)
# nx.draw(G, pos, node_size=[v * 1000 for v in flow_cent], node_color=flow_cent, cmap=plt.cm.viridis, with_labels=False)
# plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), label='Flow Centrality')
# plt.title('Flow Centrality Network Graph')
# plt.savefig('flow_centrality_graph.png')
# plt.close()


import networkx as nx
import matplotlib.pyplot as plt
import ast
import pandas as pd
import numpy as np

# Reading data
df = pd.read_csv('results/shock_results_filtered.csv')

# Adjustable parameters
YEAR_TO_VISUALIZE = 2014  # Change the desired year
NODE_SCALE_FACTOR = 50    # Set the size of the nodes
COLORMAP = plt.cm.plasma  # Change color scheme

for index, row in df.iterrows():
    if row['Year'] == YEAR_TO_VISUALIZE:
        flow_cent = ast.literal_eval(row['Flow_Cent'])

        # Create a complete graph (economic network simulation)
        G = nx.complete_graph(len(flow_cent))

        # Add flow centrality feature to nodes
        nx.set_node_attributes(
            G, {i: flow_cent[i] for i in range(len(flow_cent))}, 'flow_centrality')

        # Create an optimized layout
        pos = nx.spring_layout(
            G, k=1/np.sqrt(len(flow_cent)), iterations=50, seed=42)

        # Chart settings
        plt.figure(figsize=(16, 12))

        # Draw a network
        nodes = nx.draw_networkx_nodes(
            G, pos,
            node_size=[max(v * NODE_SCALE_FACTOR, 10)
                       for v in flow_cent],  # Minimum size
            node_color=flow_cent,
            cmap=COLORMAP,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )

        # Draw edges with transparency
        nx.draw_networkx_edges(
            G, pos,
            alpha=0.05,  # Transparency to reduce clutter
            edge_color='gray',
            width=0.3
        )

        # Add colorbar
        cbar = plt.colorbar(nodes, shrink=0.8)
        cbar.set_label('Flow Centrality', fontsize=12,
                       rotation=270, labelpad=15)

        # Title and formatting
        plt.title(
            f'Flow Centrality Network Visualization - Year {row["Year"]}\n'
            f'(SR: Before={row["Before_SR"]:.3f}, Shock={row["Shock_SR"]:.3f}, After={row["After_SR"]:.3f})',
            fontsize=14,
            pad=20
        )

        # Delete axes
        plt.axis('off')

        # Add statistical information
        stats_text = (
            f'Nodes: {len(flow_cent)}\n'
            f'Max Centrality: {max(flow_cent):.4f}\n'
            f'Min Centrality: {min(flow_cent):.4f}\n'
            f'Mean Centrality: {np.mean(flow_cent):.4f}'
        )
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="white", alpha=0.8),
                 verticalalignment='top', fontsize=10)

        plt.tight_layout()
        plt.savefig(
            f'flow_centrality_{row["Year"]}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Chart for the year {row['Year']} Saved")
        break
else:
    print(f" year {YEAR_TO_VISUALIZE} Not found in the data   ")
