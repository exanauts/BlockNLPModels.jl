"""
    DualizedNLPBlockModel{T, S} <: AbstractNLPModel{T, S}

A data type to store dualized subproblems.
"""
mutable struct DualizedNLPBlockModel{T,S} <: AbstractNLPModel{T,S}
    meta::NLPModelMeta{T,S}
    counters::Counters
    problem_block::AbstractNLPModel
    λ::AbstractVector # dual variables
    A::AbstractArray # linking matrices
end

"""
    DualizedNLPBlockModel(
        nlp::AbstractNLPModel,
        λ::AbstractVector,
        A::AbstractArray
    )

Modifies a subproblem by dualizing the linking constraints.

# Arguments
- `nlp::AbstractNLPModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
- `A::AbstractArray`: block linking matrix corresponding to the subproblem `nlp`
"""
function DualizedNLPBlockModel(nlp::AbstractNLPModel, λ::AbstractVector, A::AbstractArray)
    meta = nlp.meta
    return DualizedNLPBlockModel(meta, Counters(), nlp, λ, A)
end

"""
    update_dual!(
      nlp::DualizedNLPBlockModel,
      λ::AbstractVector,
    )
Updates the dual solution in-place for the dualized nlp block `nlp`.

# Arguments
- `nlp::DualizedNLPBlockModel`: the subproblem
- `λ::AbstractVector`: vector of dual variables
"""
function update_dual!(nlp::DualizedNLPBlockModel, λ::AbstractVector)
    nlp.λ .= λ
end

function NLPModels.obj(nlp::DualizedNLPBlockModel, x::AbstractVector)
    n = nlp.problem_block.meta.nvar
    return obj(nlp.problem_block, x) + dot(nlp.λ, nlp.A, x)
end

function NLPModels.grad!(nlp::DualizedNLPBlockModel, x::AbstractVector, g::AbstractVector)
    grad!(nlp.problem_block, x, g)
    mul!(g, nlp.A', nlp.λ, 1, 1)
    return g
end

function NLPModels.hess_structure!(
    nlp::DualizedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    return hess_structure!(nlp.problem_block, rows, cols)
end

function NLPModels.hess_coord!(
    nlp::DualizedNLPBlockModel,
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    return hess_coord!(nlp.problem_block, x, vals, obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
    nlp::DualizedNLPBlockModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
) where {T}
    return hess_coord!(nlp.problem_block, x, y, vals, obj_weight = obj_weight)
end

function NLPModels.cons!(nlp::DualizedNLPBlockModel, x::AbstractVector, cx::AbstractVector)
    return cons!(nlp.problem_block, x, cx)
end

function NLPModels.jac_structure!(
    nlp::DualizedNLPBlockModel,
    rows::AbstractVector{T},
    cols::AbstractVector{T},
) where {T}
    return jac_structure!(nlp.problem_block, rows, cols)
end

function NLPModels.jac_coord!(
    nlp::DualizedNLPBlockModel,
    x::AbstractVector,
    vals::AbstractVector,
)
    return jac_coord!(nlp.problem_block, x, vals)
end
