"""Quickstart example for the Adaptive Conductance Solver."""
from kirchhoffsolver.solver import Network, solve

# Build a small triangular network
network = Network(num_nodes=3, edges=[(0, 1), (1, 2), (0, 2)])

# Initial conductances for each edge
G0 = [1.0, 1.0, 1.0]

# Injected currents at each node
I_inj = [1.0, 0.0, -1.0]

# Run the solver for a few steps
G, V = solve(network, G0, I_inj, steps=5)

print("Final conductances:", G)
print("Node voltages:", V)
