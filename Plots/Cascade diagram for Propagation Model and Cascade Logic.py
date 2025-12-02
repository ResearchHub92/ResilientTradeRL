def _simulate_cascade(self, steps=5):
    cascade_nodes = []
    for step in range(steps):
        degrees = scatter_add(torch.ones_like(
            self.graph.edge_weight), self.graph.edge_index[0], dim=0, dim_size=NUM_NODES)
        isolated = (degrees == 0).nonzero().squeeze().tolist()
        cascade_nodes.append(isolated)
        # Simulate deletion: e.g. deleting edges associated with isolated (simplification)
        mask = ~torch.isin(self.graph.edge_index[0], torch.tensor(
            isolated, device=self.device))
        self.graph.edge_index = self.graph.edge_index[:, mask]
        self.graph.edge_weight = self.graph.edge_weight[mask]
    return cascade_nodes
# Description: This new function simulates the cascade. Call it in evaluate_shock_fully.py after the shock and append the list to the CSV. (cascade_nodes as list).
