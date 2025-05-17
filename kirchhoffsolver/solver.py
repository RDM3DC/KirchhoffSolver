"""Adaptive conductance network solver.

This module implements a minimal adaptive Kirchhoff network solver based on
nonlinear Ohm's law and an adaptive resistance update. The implementation is a
simplified version of the mathematical description in the README.
"""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass
class Network:
    """Simple electrical network specification."""

    num_nodes: int
    edges: Iterable[Tuple[int, int]]

    def incidence_matrix(self) -> np.ndarray:
        """Return the incidence matrix B with shape (E, N)."""
        edges = list(self.edges)
        m = len(edges)
        B = np.zeros((m, self.num_nodes), dtype=float)
        for idx, (i, j) in enumerate(edges):
            B[idx, i] = 1.0
            B[idx, j] = -1.0
        return B


def solve_step(G: np.ndarray, B: np.ndarray, I_inj: np.ndarray, ground: int = 0) -> np.ndarray:
    """Solve for node voltages with a fixed conductance vector.

    Parameters
    ----------
    G : array_like
        Conductance for each edge.
    B : ndarray
        Incidence matrix of shape (E, N).
    I_inj : ndarray
        Injected current at each node.
    ground : int, optional
        Index of the ground node where the voltage is fixed to zero.

    Returns
    -------
    V : ndarray
        Node voltages solving the KCL system.
    """
    m, n = B.shape
    G_diag = np.diag(G)
    G_nodal = B.T @ G_diag @ B

    # Remove ground node to solve
    mask = np.ones(n, dtype=bool)
    mask[ground] = False
    G_red = G_nodal[mask][:, mask]
    I_red = I_inj[mask]

    V_red = np.linalg.solve(G_red, I_red)
    V = np.zeros(n)
    V[mask] = V_red
    return V


def update_conductance(G: np.ndarray, currents: np.ndarray, alpha: float, mu: float, dt: float) -> np.ndarray:
    """Implicit Euler update for the adaptive conductance."""
    return (G + dt * alpha * np.abs(currents)) / (1.0 + dt * mu)


@dataclass
class AdaptiveConductanceSolver:
    """Full adaptive network solver."""

    network: Network
    alpha: float = 1.0
    mu: float = 0.1
    dt: float = 1.0

    def step(self, G: np.ndarray, I_inj: np.ndarray, ground: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Perform one implicit Euler step returning updated (G, V)."""
        B = self.network.incidence_matrix()
        V = solve_step(G, B, I_inj, ground=ground)
        currents = G * (B @ V)
        G_next = update_conductance(G, currents, self.alpha, self.mu, self.dt)
        return G_next, V


def solve(network: Network, G0: np.ndarray, I_inj: np.ndarray, steps: int, alpha: float = 1.0, mu: float = 0.1, dt: float = 1.0, ground: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Run the solver for a number of steps."""
    solver = AdaptiveConductanceSolver(network, alpha=alpha, mu=mu, dt=dt)
    G = np.asarray(G0, dtype=float)
    V = np.zeros(network.num_nodes)
    for _ in range(steps):
        G, V = solver.step(G, I_inj, ground=ground)
    return G, V
