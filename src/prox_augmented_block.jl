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
    A::AbstractArray # linking matrix
    b::AbstractVector
    sol::AbstractVector # current primal solution
    P::AbstractArray # proximal term penalty parameters
end

"""
    ProxAugmentedNLPBlockModel(
        nlp::AbstractNLPModel,
        λ::AbstractVector,
        ρ::Number,
        A::AbstractArray,
        b::AbstractVector,
        sol::AbstractVector,
        P::AbstractArray
    )
Modifies a subproblem by penalizing and dualizing the linking constraints.
# Arguments
- `nlp::AbstractNLPModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
- `ρ::Number`: penalty parameter
- `A::AbstractArray`: full linking matrix
- `b::AbstractVector`: RHS vector for linking constraints
- `sol::AbstractVector`: vector of primal variables
- `P::Vector{AbstractArray}`: matrix of proximal term's penalty parameters
"""
function ProxAugmentedNLPBlockModel(
    nlp::AbstractNLPModel,
    λ::AbstractVector,
    ρ::Number,
    A::AbstractArray,
    b::AbstractVector,
    sol::AbstractVector,
    P::AbstractArray,
)

    # Update nnzh
    H =
        sparse(
            hess_structure(nlp.problem_block)[1],
            hess_structure(nlp.problem_block)[2],
            ones(nlp.meta.nnzh),
            nlp.meta.nvar,
            nlp.meta.nvar,
        ) +
        ρ .* abs.(A[:, nlp.var_idx]' * A[:, nlp.var_idx]) +
        abs.(P)

    meta = NLPModelMeta(
        nlp.meta.nvar,
        ncon = nlp.meta.ncon,
        nnzh = nnz(H),
        nnzj = nlp.meta.nnzj,
        x0 = zeros(Float64, nlp.meta.nvar),
        lvar = nlp.meta.lvar,
        uvar = nlp.meta.uvar,
        lcon = nlp.meta.lcon,
        ucon = nlp.meta.ucon,
        minimize = true,
        name = "augmented_block",
    )
    return ProxAugmentedNLPBlockModel(meta, Counters(), nlp, λ, ρ, A, b, sol, P)
end

"""
    get_hessian(m::AugmentedNLPBlockModel;
        x::Union{AbstractVector, Nothing} = nothing,
        obj_weight::Union{AbstractVector, Nothing} = nothing,
        y::Union{AbstractVector, Nothing} = nothing,
        vals::Union{AbstractVector, Nothing} = nothing,
    )
Returns the Hessian for an augmented subproblem as a sparse matrix.
# Arguments
- `m::AugmentedNLPBlockModel`: the subproblem
- `x::Union{AbstractVector, Nothing}`: current primal solution (optional)
- `obj_weight::Union{AbstractVector, Nothing}`: objective weight
- `y::Union{AbstractVector, Nothing}`: vector of dual variables
- `vals::Union{AbstractVector, Nothing}`: nonzero values of the Hessian matrix
"""
function get_hessian(
    m::ProxAugmentedNLPBlockModel;
    x::Union{AbstractVector,Nothing} = nothing,
    y::Union{AbstractVector,Nothing} = nothing,
    obj_weight::Union{Number,Nothing} = nothing,
    vals::Union{AbstractVector,Nothing} = nothing,
)

    nlp = m.subproblem.problem_block
    H =
        sparse(
            hess_structure(nlp)[1],
            hess_structure(nlp)[2],
            ones(nlp.meta.nnzh),
            nlp.meta.nvar,
            nlp.meta.nvar,
        ) +
        m.ρ .* abs.(m.A[:, m.subproblem.var_idx]' * m.A[:, m.subproblem.var_idx]) +
        abs.(m.P)
    rows, cols, temp_vals = findnz(H)
    if isnothing(x)
        return rows, cols
    elseif isnothing(y)
        H =
            sparse(
                hess_structure(nlp)[1],
                hess_structure(nlp)[2],
                hess_coord(nlp, x, obj_weight = obj_weight),
                nlp.meta.nvar,
                nlp.meta.nvar,
            ) +
            m.ρ .* m.A[:, m.subproblem.var_idx]' * m.A[:, m.subproblem.var_idx] +
            m.P
        for i = 1:length(rows)
            vals[i] = H[rows[i], cols[i]]
        end
    else
        H =
            sparse(
                hess_structure(nlp)[1],
                hess_structure(nlp)[2],
                hess_coord(nlp, x, y, obj_weight = obj_weight),
                nlp.meta.nvar,
                nlp.meta.nvar,
            ) +
            m.ρ .* m.A[:, m.subproblem.var_idx]' * m.A[:, m.subproblem.var_idx] +
            m.P
        for i = 1:length(rows)
            vals[i] = H[rows[i], cols[i]]
        end
    end
    return H
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
function update_primal!(nlp::ProxAugmentedNLPBlockModel, sol::AbstractVector)
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
function update_dual!(nlp::ProxAugmentedNLPBlockModel, λ::AbstractVector)
    nlp.λ .= λ
end

function NLPModels.obj(nlp::ProxAugmentedNLPBlockModel, x::AbstractVector)
    local_sol = deepcopy(nlp.sol)
    local_sol[nlp.subproblem.var_idx] = x

    return obj(nlp.subproblem.problem_block, x) +
           dot(nlp.λ, nlp.A[:, nlp.subproblem.var_idx], x) +
           (nlp.ρ / 2) * norm(nlp.A * local_sol - nlp.b)^2 +
           1 / 2 * dot(
               (x - nlp.sol[nlp.subproblem.var_idx]),
               nlp.P,
               (x - nlp.sol[nlp.subproblem.var_idx]),
           )
end

function NLPModels.grad!(nlp::ProxAugmentedNLPBlockModel, x::AbstractVector, g::AbstractVector)
    local_sol = deepcopy(nlp.sol)
    local_sol[nlp.subproblem.var_idx] = x

    grad!(nlp.subproblem.problem_block, x, g)
    g .+=
        nlp.A[:, nlp.subproblem.var_idx]' * nlp.λ +
        nlp.ρ .* nlp.A[:, nlp.subproblem.var_idx]' * (nlp.A * local_sol - nlp.b) +
        nlp.P * (x - nlp.sol[nlp.subproblem.var_idx])
    return g
end

function NLPModels.hess_structure!(
    nlp::ProxAugmentedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    temp_r, temp_c = get_hessian(nlp)
    rows .= temp_r
    cols .= temp_c
    return rows, cols
end

function NLPModels.hess_coord!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    get_hessian(nlp, x = x, obj_weight = obj_weight, vals = vals)
    return vals
end

function NLPModels.hess_coord!(
    nlp::ProxAugmentedNLPBlockModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    get_hessian(nlp, x = x, y = y, obj_weight = obj_weight, vals = vals)
    return vals
end

function NLPModels.cons!(nlp::ProxAugmentedNLPBlockModel, x::AbstractVector, cx::AbstractVector)
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
