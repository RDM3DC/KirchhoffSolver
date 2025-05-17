# KirchhoffSolver

## Adaptive Conductance Solver — Mathematical Foundations

We model a resistive network with **adaptive, time-evolving conductances** governed by the **Adaptive Resistance Principle (ARP)**. This system generalizes Ohm’s Law and Kirchhoff’s Laws into a differentiable and stable dynamical framework.

## Installation

To install the solver directly from the repository, clone the project and run:

```bash
python -m pip install .
```


### 1. Ohm’s Law (Generalized)

For each branch $(i,j)$ in the circuit:

$$
I_{ij}(t) = f(V_i(t) - V_j(t); \theta)
$$

For linear conductors:

$$
f(\Delta V; G_{ij}) = G_{ij} \cdot \Delta V
$$

For exotic materials (e.g. memristors, VO₂, NDR):

$$
f(\Delta V; \theta) = a \tanh(b\Delta V) + c \cdot \text{sign}(\Delta V) \cdot |\Delta V|^d
$$

where $\theta = [a, b, c, d]$ are material parameters.

### 2. Kirchhoff’s Current Law (KCL)

At every internal node $k$, the net current is zero:

$$
\sum_j I_{kj}(t) = 0
$$

Given conductances $G_{ij}(t)$, and nodal voltages $V(t) \in \mathbb{R}^N$, define the **nodal conductance matrix**:

$$
G_{\text{nodal}} = B^\top \cdot \text{diag}(G) \cdot B
$$

Then the KCL system becomes:

$$
G_{\text{nodal}} V = I_{\text{inj}}
$$

Where $B \in \mathbb{R}^{E \times N}$ is the incidence matrix of the graph.

### 3. ARP: Adaptive Resistance Principle

Each branch’s conductance evolves according to:

$$
\frac{dG_{ij}}{dt} = \alpha \cdot |I_{ij}(t)| - \mu \cdot G_{ij}(t)
$$

This reflects:

* **Positive reinforcement** of high-current paths
* **Exponential decay** of unused paths

### 4. Lyapunov Energy

Define:

$$
\mathcal{E}(G) = \frac{\mu}{2\alpha} \sum_{ij} G_{ij}^2 - \sum_{ij} \int_0^{G_{ij}} |I_{ij}(s)| ds
$$

Then:

$$
\frac{d\mathcal{E}}{dt} = -\alpha \cdot \left\| \nabla \mathcal{E}(G) \right\|^2 \le 0
$$

This guarantees:

* **Global stability**
* **Unique equilibrium**
* **Convergence to** $G^* \text{ such that } |I_{ij}^*| = \frac{\mu}{\alpha} G_{ij}^*$

### 5. Implicit Euler Integration

The adaptive update is solved with:

$$
G^{k+1} = G^k + \Delta t ( \alpha |I(G^{k+1})| - \mu G^{k+1} )
$$

Which is equivalent to minimizing:

$$
G^{k+1} = \arg\min_G \left\{ \mathcal{E}(G) + \frac{\|G - G^k\|^2}{2 \Delta t / \alpha} \right\}
$$

This is a **proximal-gradient step**, ensuring unconditional stability.

### 6. Differentiability

Both solvers support automatic differentiation via:

* **Forward-mode JAX:** for simulation
* **Custom VJP / backward pass:** to enable backprop through `G \mapsto I(G)` using the **Implicit Function Theorem**

Let:

$$
F(G) = G - G^k - \Delta t ( \alpha |I(G)| - \mu G )
$$

Then the adjoint gradient solves:

$$
(J_F)^\top \cdot \lambda = \frac{\partial L}{\partial G} \quad\Rightarrow\quad \frac{\partial L}{\partial G^k} = -\lambda
$$

### Summary

* Solves **nonlinear**, time-evolving Kirchhoff networks
* Stable for arbitrary time-steps
* Scales to ≥1 million branches
* Fully differentiable (JAX)
* Supports exotic devices: memristors, diodes, superconductors
* Guarantees convergence through Lyapunov energy

## Quickstart Example

Run `python examples/quickstart.py` to simulate a small three-node network.
The script shows how to construct a network, run the solver and print
final conductances and node voltages.

## Documentation

API documentation can be generated with [Sphinx](https://www.sphinx-doc.org/):

```bash
cd docs && sphinx-build -b html . _build
```

The solver and helper functions contain comprehensive docstrings that
feed directly into the documentation.

## Performance Notes

The solver uses an implicit integration scheme which is stable for large
time steps. For networks with millions of branches, performance scales
roughly linearly with the number of edges thanks to the sparse updates.

