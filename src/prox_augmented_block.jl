"""
    ProxAugmentedNLPBlockModel{T, S} <: AbstractNLPModel{T, S}

A data type to store augmented subproblems.
"""
mutable struct ProxAugmentedNLPBlockModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    subproblem::AbstractNLPModel
    λ::AbstractVector # dual variables
    ρ::Number # penalty Parameter
    A::AbstractMatrix # linking matrix
    b::AbstractVector
    sol::AbstractVector # current primal solution
    block_sol::AbstractVector # current block primal solution for the proximal term
    hess_info::AugmentedHessianInfo # precompute to save computation effort
    P::AbstractMatrix # proximal term's penalty parameters
end

"""
    ProxAugmentedNLPBlockModel(
        nlp::AbstractNLPModel,
        λ::AbstractVector,
        ρ::Number,
        A::AbstractMatrix,
        b::AbstractVector,
        sol::AbstractVector,
        P::AbstractMatrix
    )

Modifies a subproblem by penalizing and dualizing the linking constraints.

# Arguments
- `nlp::AbstractNLPModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
- `ρ::Number`: penalty parameter
- `A::AbstractMatrix`: full linking matrix
- `b::AbstractVector`: RHS vector for linking constraints
- `sol::AbstractVector`: vector of primal variables
- `P::AbstractMatrix`: matrix of proximal term's penalty parameters
"""
function ProxAugmentedNLPBlockModel(
    nlp::AbstractNLPModel,
    λ::AbstractVector,
    ρ::Number,
    A::AbstractMatrix,
    b::AbstractVector,
    sol::AbstractVector,
    P::AbstractMatrix,
)
    if ρ != 0
        ATA = findnz(A[:, nlp.var_idx]' * A[:, nlp.var_idx])
    else
        ATA = ([], [], [])
    end
    block_hess_struct = hess_structure(nlp.problem_block)
    P_triplets = findnz(P)

    I = vcat(block_hess_struct[1], ATA[1], P_triplets[1])
    J = vcat(block_hess_struct[2], ATA[2], P_triplets[2])
    V = vcat(ones(nlp.meta.nnzh), ρ .* abs.(ATA[3]), abs.(P_triplets[3]))
    prox_aug_hess_struct = findnz(sparse(I, J, V, nlp.meta.nvar, nlp.meta.nvar))
    meta = update_nnzh(nlp.meta, length(prox_aug_hess_struct[1]))

    hess_struct = AugmentedHessianInfo(
        (prox_aug_hess_struct[1], prox_aug_hess_struct[2]),
        (block_hess_struct[1], block_hess_struct[2], sortperm(block_hess_struct[1])),
        ATA,
    )
    return ProxAugmentedNLPBlockModel(
        meta,
        Counters(),
        nlp,
        λ,
        ρ,
        A,
        b,
        deepcopy(sol),
        sol[nlp.var_idx], # for the proximal term
        hess_struct,
        P,
    )
end

"""
    update_primal!(
        nlp::ProxAugmentedNLPBlockModel,
        sol::AbstractVector
    )

Updates the primal solution estimate for the augmented nlp block `nlp`.

# Arguments
- `nlp::ProxAugmentedNLPBlockModel`: the subproblem
- `sol::AbstractVector`: vector of primal variables
"""
function update_primal!(nlp::ProxAugmentedNLPBlockModel, sol::AbstractVector)
    nlp.sol .= sol
    nlp.block_sol .= sol[nlp.subproblem.var_idx]
end

"""
    update_dual!(
        nlp::ProxAugmentedNLPBlockModel,
        λ::AbstractVector,
    )

Updates the dual solution estimate in-place for the augmented nlp block `nlp`.
# Arguments
- `nlp::ProxAugmentedNLPBlockModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
"""
function update_dual!(nlp::ProxAugmentedNLPBlockModel, λ::AbstractVector)
    nlp.λ .= λ
end

"""
    update_rho!(
        nlp::ProxAugmentedNLPBlockModel,
        ρ::Number,
    )

Updates the penalty parameter in-place for the proximal augmented nlp block `nlp`.

# Arguments
- `nlp::ProxAugmentedNLPBlockModel`: the subproblem
- `ρ::Number`: vector of dual variables
"""
function update_rho!(nlp::ProxAugmentedNLPBlockModel, ρ::Number)
    nlp.ρ = ρ
end

function NLPModels.obj(nlp::ProxAugmentedNLPBlockModel, x::AbstractVector)
    nlp.sol[nlp.subproblem.var_idx] = x
    return obj(nlp.subproblem.problem_block, x) +
           dot(nlp.λ, nlp.A[:, nlp.subproblem.var_idx], x) +
           (nlp.ρ / 2) * norm(nlp.A * nlp.sol - nlp.b)^2 +
           1 / 2 * dot(
               (x - nlp.block_sol),
               nlp.P,
               (x - nlp.block_sol),
           )
end

function NLPModels.grad!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector,
    g::AbstractVector,
)
    nlp.sol[nlp.subproblem.var_idx] = x

    grad!(nlp.subproblem.problem_block, x, g)
    mul!(
        g,
        nlp.A[:, nlp.subproblem.var_idx]',
        (nlp.λ + nlp.ρ .* (nlp.A * nlp.sol - nlp.b)),
        1,
        1,
    )
    mul!(g, nlp.P, (x - nlp.block_sol), 1, 1)
    return g
end

function NLPModels.hess_structure!(
    nlp::ProxAugmentedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    rows .= nlp.hess_info.augmented_hessian_struct[1]
    cols .= nlp.hess_info.augmented_hessian_struct[2]
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    get_augmented_hessian_coord!(nlp, nlp.ρ, x, vals, obj_weight)

    aug_hess = nlp.hess_info.augmented_hessian_struct
    P_triplets = findnz(nlp.P)
    # add the proximal term weights
    main_idx = 1
    sub_idx = 1
    for (i, j) in zip(aug_hess[1], aug_hess[2])
        if P_triplets[1][sub_idx] == i && P_triplets[2][sub_idx] == j
            vals[main_idx] += P_triplets[3][sub_idx]
            sub_idx += 1
        end
        main_idx += 1
    end
    return vals
end

function NLPModels.hess_coord!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    get_augmented_hessian_coord!(nlp, nlp.ρ, x, vals, obj_weight, y = y)

    aug_hess = nlp.hess_info.augmented_hessian_struct
    P_triplets = findnz(nlp.P)
    # add the proximal term weights
    main_idx = 1
    sub_idx = 1
    for (i, j) in zip(aug_hess[1], aug_hess[2])
        if P_triplets[1][sub_idx] == i && P_triplets[2][sub_idx] == j
            vals[main_idx] += P_triplets[3][sub_idx]
            sub_idx += 1
        end
        main_idx += 1
    end
    return vals
    return vals
end

function NLPModels.cons!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector,
    cx::AbstractVector,
)
    return cons!(nlp.subproblem.problem_block, x, cx)
end

function NLPModels.jac_structure!(
    nlp::ProxAugmentedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    return jac_structure!(nlp.subproblem.problem_block, rows, cols)
end

function NLPModels.jac_coord!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    return jac_coord!(nlp.subproblem.problem_block, x, vals)
end
