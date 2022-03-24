module BlockNLPModels

using NLPModels

export BlockNLPModel
export AbstractBlockNLP

abstract type AbstractBlockNLP end

mutable struct BlockNLPModel{T<:AbstractNLPModel} <: AbstractBlockNLP
    blocks::Vector{T}
    linkconstraints::Vector{Float64}
    λ::Float64
  end

function BlockNLPModel(blocks::Vector{AbstractNLPModel}, coeff::Vector{S}; λ0::Union{Nothing, S} = nothing) where {S}
    # Initialize dual variable
    λ0 === nothing ? λ = 0.0 : λ = λ0
    return BlockNLPModel(blocks, coeff, λ)
end


end # module