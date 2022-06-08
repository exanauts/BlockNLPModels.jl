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

In addition to a modeling framework, this package also provides several callback functions to enable easy implementation of several popular NLP decomposition algorithms to efficiently solve large instances of block structured NLPs. We highlight some of the important ones here:

1. For the dual decomposition method, one can call `DualizedNLPBlockModel` for the subproblems to dualize their objective. This method is defined as follows:

    `DualizedNLPBlockModel(nlp, λ, A)`, where
    - nlp: the subproblem to be dualized as an instance of `AbstractNLPModel`
    - λ: vector of dual variables corresponding to the linking constraints assosciated with the nlp block `nlp`
    - A: linking constraint matrix block assosciated with the nlp block `nlp`

This function will return the dualized block model as an instance of `AbstractNLPModel`.

As an example, suppose for the problem shown above, we wish to dualize the second nlp block. Here we assume that the dual variables are stored in variable `y`. Dualization can be achieved as follows:
```julia
A = get_linking_matrix_blocks(blocknlp)
dualized_block = DualizedNLPBlockModel(blocknlp.blocks[2].problem_block, y[blocknlp.problem_size.con_counter+1:end], A[2])
```

2. To support the implementation of ADMM, we provide the following callback function:

`AugmentedNLPBlockModel(nlp, y, ρ, A, b, x)`, where
- nlp: the subproblem which is to be augmented
- y: the dual variables corresponding to the linking constraints
- $\rho$: penalty parameter
- A: matrix containing the linking constraint coefficients
- b: the right-hand-side vector for the linking constraints
- x: current estimate of the primal variables

The output of this function will be the augmented block model as an instance of `AbstractNLPModel`.

Again, as an illustration, we augment the objective function of the secoond NLP block of the above example problem. We assume `x` and `y` are the current estimates of the primal and dual variables, respectively. Further, we assume `ρ` to be the penalty parameter.

```julia
A = get_linking_matrix(blocknlp)
b = get_rhs_vector(blocknlp)
dualized_block = AugmentedNLPBlockModel(blocknlp.blocks[2], y[blocknlp.problem_size.con_counter+1:end], ρ, A, b, x)
```

3. We also support the addition of a proximal term to an augmented objective. This can be achieved with the following method:

`ProxAugmentedNLPBlockModel(nlp, y, ρ, A, b, x, P)`, where `P`$\, \in \mathbb{R}^{m_i \times m_i}$ is a sparse matrix containing the weights for the proximal term. All other arguments are the same as defined for the `AugmentedNLPBlockModel` method.

This implementation of this method follows the same steps as for the `AugmentedNLPBlockModel` method. The only difference being the addition of a sparse `P` matrix which can be defined by making use of the `SparseArrays.jl` package in Julia.

4. Finally, our package also provides the user an option to convert the `AbstractBlockNLPModel` into a full-space `AbstractNLPModel` which can be solved with any general purpose NLP solver. The method for this is as follows:

`FullSpaceModel(block_nlp_model)`, where `block_nlp_model` is an instance of `AbstractBlockNLPModel`.

For example, suppose we desire to solve the three block problem described above with `MadNLP.jl`. This can be done by executing the following set of commands:
```julia
full_model = FullSpaceModel(blocknlp)
solution = madnlp(full_model)
```
assuming that `MadNLP.jl` has already been imported in the environment.

## Acknowledgements
This package's development was supported by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.
