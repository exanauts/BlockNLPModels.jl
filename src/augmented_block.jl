"""
    AugmentedNLPBlockModel{T, S} <: AbstractNLPModel{T, S}

A data type to store augmented subproblems.
"""
mutable struct AugmentedNLPBlockModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    subproblem::AbstractNLPModel
    λ::AbstractVector # dual variables
    ρ::Number # penalty Parameter
    A::AbstractMatrix # linking matrix
    b::AbstractVector
    sol::AbstractVector # current primal solution
    hess_info::AugmentedHessianInfo # precompute to save computation effort
end

"""
    AugmentedNLPBlockModel(
        nlp::AbstractNLPModel,
        λ::AbstractVector,
        ρ::Number,
        A::AbstractMatrix,
        b::AbstractVector,
        sol::AbstractVector
    )

Modifies a subproblem by penalizing and dualizing the linking constraints.

# Arguments
- `nlp::AbstractNLPModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
- `ρ::Number`: penalty parameter
- `A::AbstractMatrix`: full linking matrix
- `b::AbstractVector`: RHS vector for linking constraints
- `sol::AbstractVector`: vector of primal variables
"""
function AugmentedNLPBlockModel(
    nlp::AbstractNLPModel,
    λ::AbstractVector,
    ρ::Number,
    A::AbstractMatrix,
    b::AbstractVector,
    sol::AbstractVector,
)
    if ρ != 0
        ATA = findnz(A[:, nlp.var_idx]' * A[:, nlp.var_idx])
    else
        ATA = ([], [], [])
    end

    block_hess_struct = hess_structure(nlp.problem_block)
    I = vcat(block_hess_struct[1], ATA[1])
    J = vcat(block_hess_struct[2], ATA[2])
    V = vcat(ones(nlp.meta.nnzh), ρ .* abs.(ATA[3]))
    aug_hess_struct = findnz(sparse(I, J, V, nlp.meta.nvar, nlp.meta.nvar))
    meta = update_nnzh(nlp.meta, length(aug_hess_struct[1]))

    hess_struct = AugmentedHessianInfo(
        (aug_hess_struct[1], aug_hess_struct[2]),
        (block_hess_struct[1], block_hess_struct[2], sortperm(block_hess_struct[1])),
        ATA,
    )
    return AugmentedNLPBlockModel(meta, Counters(), nlp, λ, ρ, A, b, deepcopy(sol), hess_struct)
end

"""
    update_primal!(
        nlp::AugmentedNLPBlockModel,
        sol::AbstractVector
    )

Updates the primal solution estimate for the augmented nlp block `nlp`.

# Arguments
- `nlp::AugmentedNLPBlockModel`: the subproblem
- `sol::AbstractVector`: vector of primal variables
"""
function update_primal!(nlp::AugmentedNLPBlockModel, sol::AbstractVector)
    nlp.sol .= sol
end

"""
    update_dual!(
      nlp::AugmentedNLPBlockModel,
      λ::AbstractVector,
    )

Updates the dual solution estimate in-place for the augmented nlp block `nlp`.

# Arguments
- `nlp::AugmentedNLPBlockModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
"""
function update_dual!(nlp::AugmentedNLPBlockModel, λ::AbstractVector)
    nlp.λ .= λ
end

"""
    update_rho!(
      nlp::AugmentedNLPBlockModel,
      ρ::Number,
    )

Updates the penalty parameter in-place for the augmented nlp block `nlp`.
# Arguments
- `nlp::AugmentedNLPBlockModel`: the subproblem
- `ρ::Number`: vector of dual variables
"""
function update_rho!(nlp::AugmentedNLPBlockModel, ρ::Number)
    nlp.ρ = ρ
end

function NLPModels.obj(nlp::AugmentedNLPBlockModel, x::AbstractVector)
    nlp.sol[nlp.subproblem.var_idx] = x

    return obj(nlp.subproblem.problem_block, x) +
           dot(nlp.λ, nlp.A[:, nlp.subproblem.var_idx], x) +
           (nlp.ρ / 2) * norm(nlp.A * nlp.sol - nlp.b)^2
end

function NLPModels.grad!(nlp::AugmentedNLPBlockModel, x::AbstractVector, g::AbstractVector)
    nlp.sol[nlp.subproblem.var_idx] = x

    grad!(nlp.subproblem.problem_block, x, g)
    mul!(
        g,
        nlp.A[:, nlp.subproblem.var_idx]',
        nlp.λ + nlp.ρ .* (nlp.A * nlp.sol - nlp.b),
        1,
        1,
    )
    return g
end

function NLPModels.hess_structure!(
    nlp::AugmentedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    rows .= nlp.hess_info.augmented_hessian_struct[1]
    cols .= nlp.hess_info.augmented_hessian_struct[2]
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::AugmentedNLPBlockModel,
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    get_augmented_hessian_coord!(nlp, nlp.ρ, x, vals, obj_weight)
    return vals
end

function NLPModels.hess_coord!(
    nlp::AugmentedNLPBlockModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    get_augmented_hessian_coord!(nlp, nlp.ρ, x, vals, obj_weight, y = y)
    return vals
end

function NLPModels.cons!(nlp::AugmentedNLPBlockModel, x::AbstractVector, cx::AbstractVector)
    return cons!(nlp.subproblem.problem_block, x, cx)
end

function NLPModels.jac_structure!(
    nlp::AugmentedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    return jac_structure!(nlp.subproblem.problem_block, rows, cols)
end

function NLPModels.jac_coord!(
    nlp::AugmentedNLPBlockModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    return jac_coord!(nlp.subproblem.problem_block, x, vals)
end
