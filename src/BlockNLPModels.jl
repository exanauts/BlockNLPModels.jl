module BlockNLPModels

using NLPModels
using LinearAlgebra

export AbstractBlockNLP
export BlockNLPModel
export DualizedNLPblockModel
export FullSpaceModel

abstract type AbstractBlockNLP end

mutable struct BlockNLPModel{T<:AbstractNLPModel} <: AbstractBlockNLP
  blocks::Vector{T}
  linkconstraints::Vector{Float64}
end

mutable struct DualizedNLPblockModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  problem_block::AbstractNLPModel
  λ::AbstractVector # dual variables
  a::AbstractVector # linking constraint coeff
end

function DualizedNLPblockModel(nlp::AbstractNLPModel, λ::AbstractVector, a::AbstractVector)
  meta = nlp.meta
  return DualizedNLPblockModel(meta, Counters(), nlp, λ, a)
end

function NLPModels.obj(nlp::DualizedNLPblockModel, x::AbstractVector)
  n = nlp.problem_block.meta.nvar
  d(x) = dot(nlp.λ, dot(nlp.a, x))
  return obj(nlp.problem_block, x) + d(x)
end

function NLPModels.grad!(nlp::DualizedNLPblockModel, x::AbstractVector, g::AbstractVector)
  grad!(nlp.problem_block, x, g)
  g .+= dot(nlp.λ, nlp.a)
  return g
end

function NLPModels.hess_structure!(nlp::DualizedNLPblockModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
  return hess_structure!(nlp.problem_block, rows, cols)
end

function NLPModels.hess_coord!(
  nlp::DualizedNLPblockModel,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
  return hess_coord!(nlp.problem_block, x, vals, obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  nlp::DualizedNLPblockModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
  ) where {T}
  return hess_coord!(nlp.problem_block, x, y, vals, obj_weight = obj_weight)
end

function NLPModels.cons!(nlp::DualizedNLPblockModel, x::AbstractVector, cx::AbstractVector)
  return cons!(nlp.problem_block, x, cx)
end

function NLPModels.jac_structure!(nlp::DualizedNLPblockModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
  return jac.structure!(nlp.problem_block, rows, cols)
end

function NLPModels.jac_coord!(nlp::DualizedNLPblockModel, x::AbstractVector, vals::AbstractVector)
  return jac_coord!(nlp.problem_block, x, vals)
end

include("full_space.jl")
end # module