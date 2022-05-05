"""
    AugmentedNLPBlockModel{T, S} <: AbstractNLPModel{T, S}

A data type to store augmented subproblems.
"""
mutable struct AugmentedNLPBlockModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    subproblem::AbstractNLPModel
    λ::AbstractVector # dual variables
    ρ::Number # penalty Parameter
    A::AbstractArray # linking matrix
    b::AbstractVector
    sol::AbstractVector # current primal solution
end

"""
    AugmentedNLPBlockModel(
        nlp::AbstractNLPModel,
        λ::AbstractVector,
        ρ::Number,
        A::AbstractArray,
        b::AbstractVector,
        sol::AbstractVector
    )
Modifies a subproblem by penalizing and dualizing the linking constraints.

# Arguments
- `nlp::AbstractNLPModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
- `ρ::Number`: penalty parameter
- `A::AbstractArray`: full linking matrix
- `b::AbstractVector`: RHS vector for linking constraints
- `sol::AbstractVector`: vector of primal variables
"""
function AugmentedNLPBlockModel(nlp::AbstractNLPModel, λ::AbstractVector,
        ρ::Number, A::AbstractArray, b::AbstractVector, sol::AbstractVector)

    # Update nnzh
    H = sparse(
               hess_structure(nlp.problem_block)[1],
               hess_structure(nlp.problem_block)[2],
               ones(nlp.meta.nnzh),
               nlp.meta.nvar,
               nlp.meta.nvar
    ) + ρ.*abs.(A[:, nlp.var_idx]'*A[:, nlp.var_idx])

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
    return AugmentedNLPBlockModel(meta, Counters(), nlp, λ, ρ, A, b, sol)
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
function get_hessian(m::AugmentedNLPBlockModel;
    x::Union{AbstractVector, Nothing} = nothing,
    y::Union{AbstractVector, Nothing} = nothing,
    obj_weight::Union{Number, Nothing} = nothing,
    vals::Union{AbstractVector, Nothing} = nothing,
)

    nlp = m.subproblem.problem_block
    H = sparse(
               hess_structure(nlp)[1],
               hess_structure(nlp)[2],
               ones(nlp.meta.nnzh),
               nlp.meta.nvar,
               nlp.meta.nvar
              ) + m.ρ.*abs.(m.A[:, m.subproblem.var_idx]'*m.A[:, m.subproblem.var_idx])
    rows, cols, temp_vals = findnz(H)
    if isnothing(x)
        return rows, cols
    elseif isnothing(y)
        H = sparse(
                   hess_structure(nlp)[1],
                   hess_structure(nlp)[2],
                   hess_coord(nlp, x, obj_weight = obj_weight),
                   nlp.meta.nvar,
                   nlp.meta.nvar
                  ) + m.ρ.*m.A[:, m.subproblem.var_idx]'*m.A[:, m.subproblem.var_idx]
        for i in 1:length(rows)
            vals[i] = H[rows[i], cols[i]]
        end
    else
        H = sparse(
                   hess_structure(nlp)[1],
                   hess_structure(nlp)[2],
                   hess_coord(nlp, x, y, obj_weight = obj_weight),
                   nlp.meta.nvar,
                   nlp.meta.nvar
                  ) + m.ρ.*m.A[:, m.subproblem.var_idx]'*m.A[:, m.subproblem.var_idx]
        for i in 1:length(rows)
            vals[i] = H[rows[i], cols[i]]
        end
    end
    return H
end

"""
    update_primal_sol!(
      nlp::AbstractNLPModel, 
      sol::AbstractVector
    )
Updates the primal solution for the augmented nlp block `nlp`.

# Arguments
- `nlp::AbstractNLPModel`: the subproblem 
- `sol::AbstractVector`: vector of primal variables
"""
function update_primal_sol!(nlp::AugmentedNLPBlockModel, sol::AbstractVector)
    nlp.sol .= sol
end

"""
    update_dual_sol!(
      nlp::AbstractNLPModel, 
      λ::AbstractVector, 
    )
Updates the dual solution in-place for the augmented nlp block `nlp`.

# Arguments
- `nlp::AbstractNLPModel`: the subproblem 
- `λ::AbstractVector`: vector of dual variables
"""
function update_dual_sol!(nlp::AugmentedNLPBlockModel, λ::AbstractVector)
    nlp.λ .= λ
end

function NLPModels.obj(nlp::AugmentedNLPBlockModel, x::AbstractVector)
    local_sol = deepcopy(nlp.sol)
    local_sol[nlp.subproblem.var_idx] = x

    return obj(nlp.subproblem.problem_block, x) +
    dot(nlp.λ, nlp.A[:, nlp.subproblem.var_idx], x) +
    (nlp.ρ/2)*norm(nlp.A*local_sol - nlp.b)^2
end

function NLPModels.grad!(nlp::AugmentedNLPBlockModel, x::AbstractVector, g::AbstractVector)
    local_sol = deepcopy(nlp.sol)
    local_sol[nlp.subproblem.var_idx] = x

    grad!(nlp.subproblem.problem_block, x, g)
    g .+= nlp.A[:, nlp.subproblem.var_idx]'*nlp.λ +
    nlp.ρ.*nlp.A[:, nlp.subproblem.var_idx]'*(nlp.A*local_sol - nlp.b)
    return g
end

function NLPModels.hess_structure!(nlp::AugmentedNLPBlockModel,
    rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    temp_r, temp_c = get_hessian(nlp)
    rows .= temp_r
    cols .= temp_c
    return rows, cols
end

function NLPModels.hess_coord!(
        nlp::AugmentedNLPBlockModel,
        x::AbstractVector{T},
        vals::AbstractVector{T};
        obj_weight = one(T),
) where {T}
    get_hessian(nlp, x = x, obj_weight = obj_weight, vals = vals)
    return vals
end

function NLPModels.hess_coord!(
        nlp::AugmentedNLPBlockModel,
        x::AbstractVector{T},
        y::AbstractVector{T},
        vals::AbstractVector{T};
        obj_weight = one(T),
) where {T}
    get_hessian(nlp, x = x, y = y, obj_weight = obj_weight, vals = vals)
    return vals
end

function NLPModels.cons!(
    nlp::AugmentedNLPBlockModel,
    x::AbstractVector, cx::AbstractVector,
)
    return cons!(nlp.subproblem.problem_block, x, cx)
end

function NLPModels.jac_structure!(nlp::AugmentedNLPBlockModel,
    rows::AbstractVector{T}, cols::AbstractVector{T},
) where {T}
    return jac_structure!(nlp.subproblem.problem_block, rows, cols)
end

function NLPModels.jac_coord!(nlp::AugmentedNLPBlockModel,
    x::AbstractVector, vals::AbstractVector,
)
    return jac_coord!(nlp.subproblem.problem_block, x, vals)
end
