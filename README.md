# BlockNLPModels.jl
[![workflow](https://github.com/Exanauts/BlockNLPModels.jl/actions/workflows/ci.yml/badge.svg?token=2K0LJ6YJD1)](https://codecov.io/gh/exanauts/BlockNLPModels.jl)
[![codecov](https://codecov.io/gh/exanauts/BlockNLPModels.jl/branch/main/graph/badge.svg?token=2K0LJ6YJD1)](https://codecov.io/gh/exanauts/BlockNLPModels.jl)

This package provides a modeling framework based on [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) for block structured nonlinear optimization problems (NLPs) in Julia. Specifically, one can use this package to model NLPs of the following form:

$$
\begin{aligned}
  \min_{x \in \mathbb{R^{m_i}}} \quad & \sum\limits_{i \in \mathcal{B}} f_i(x_i) \\
  \mathrm{subject \, to} \quad & c_{ij} (x_i) \leq 0 \quad \forall \, i \in \mathcal{B}, j \in \mathcal{C}_i \\
  & \sum\limits_{i \in \mathcal{B}} A_i x_i = b,
\end{aligned}
$$

where $\mathcal{B}$ is the set of variable blocks and $\mathcal{C}_i$ is the set of constraints for block $i$. Functions $f_i$ and $c_{ij}$ are assumed to be generally nonlinear noconvex functions of variables assosciated with the block $i$. Further, we assume that the constraints linking the NLP blocks in set $\mathcal{B}$ are linear in the decision variables.

In this package, block structured NLPs of the above form are represented by an instance of `AbstractBlockNLPModel`.

## Installation
This package is currently under active development and has not been registered yet. However, to access the current code, one can enter the following command in Julia REPL:

```julia
]add "https://github.com/exanauts/BlockNLPModels.jl"
```

To confirm if the package was installed correctly, please run:
```julia
test BlockNLPModels
```
The test code generates and solves a small instance of a `BlockNLPModel` using the interior-point solver [MadNLP.jl](https://github.com/MadNLP/MadNLP.jl).

## Quickstart Guide
Suppose we want to model the following NLP that can be decomposed into three univariate blocks:
$$
\begin{aligned}
  \min_{x \in \mathbb{R}_+} \quad & \sum\limits_{i \in \{ 1, 2, 3 \}} (x_i - a_i)^2 \\
  \mathrm{subject \, to} \quad & x_i \leq 1 \quad \forall \, i \in \{ 1, 2, 3 \} \\
  & \sum\limits_{i \in \{ 1, 2, 3 \}} x_i = b,
\end{aligned}
$$
where $a_i$ for all i and b are some constants.

We start by initializing an empty `BlockNLPModel`:

```julia
blocknlp = BlockNLPModel()
```

Next, we add the NLP blocks to `blocknlp`. For this step, the subproblems need to be made available as `AbstractNLPModel` objects. Assuming that `nlp_blocks` is a 3-element vector containing the three NLP blocks of our example above, we add these blocks to `blocknlp` by using the `add_block` function as follows:

```julia
for i in 1:3
    add_block(blocknlp, nlp_blocks[i])
end
```

Once the blocks have been added the next step is to add the linking constraints. For this, we make use of the `add_links` method in this package. This is defined as follows:

`add_links(block_nlp_model, n_constraints, links, rhs_constants)`, where
- `block_nlp_model` is an instance of `BlockNLPModel`
- `n_constraints` is the number of linking constraints that are being added to `block_nlp_model`
- `links` is a dictionary that specifies which block is being linked with what coefficients  
- `rhs_constants` is the right-hand-side vector for the linking constraints

For our example above, we add the linking constraint as follows:
```julia
add_links(blocknlp, 1, Dict(1 => [1.], 2 => [1.], 3 => [1.]), b)
```
With this, we have finished modeling the above example as an `AbstractBlockNLPModel`.

In addition to a modeling framework, this package also provides several callback functions to enable easy implementation of several popular NLP decomposition algorithms to efficiently solve large instances of block structured NLPs. We highlight some examples here:

1. For the dual decomposition method, one can call `DualizedNLPBlockModel` for the subproblems to dualize their objective. This method is defined as follows:

    DualizedNLPBlockModel(nlp, λ, A), where
    - nlp: the subproblem to be dualized as an instance of `AbstractNLPModel`
    - λ: vector of dual variables corresponding to the linking constraints assosciated with the nlp block `nlp`
    - A: linking constraint matrix block assosciated with the nlp block `nlp`

This function will return the dualized block model as an instance of `AbstractNLPModel`.

As an example, suppose for the problem shown above, we wish to dualize the second nlp block. This can be achieved as follows:
```julia
A = get_linking_matrix_blocks(blocknlp)
dualized_block = DualizedNLPBlockModel(blocknlp.blocks[2].problem_block, y, A[2])
```
## Acknowledgements
This package's development was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
