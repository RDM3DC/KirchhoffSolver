import numpy as np
from kirchhoffsolver.solver import Network, solve_step, update_conductance


def test_incidence_matrix():
    net = Network(num_nodes=3, edges=[(0, 1), (1, 2)])
    B = net.incidence_matrix()
    assert B.shape == (2, 3)
    assert np.allclose(B[0], [1.0, -1.0, 0.0])


def test_update_conductance():
    G = np.array([1.0])
    currents = np.array([2.0])
    updated = update_conductance(G, currents, alpha=1.0, mu=0.1, dt=1.0)
    expected = (G + currents) / 1.1
    assert np.allclose(updated, expected)
