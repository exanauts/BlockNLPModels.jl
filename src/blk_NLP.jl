abstract type BlockNLP end
mutable struct BlockNLPModel{T} <: BlockNLP
    blocks::Vector{T}
    linkconstraints::Vector{Float64}
    λ::Float64
  end

function BlockNLPModel(blocks::Vector{T}, coeff::Vector{S}; λ0::Union{Nothing, S}) where {T, S}
    # Initialize dual variable
    λ0 === nothing ? λ = 0.0 : λ = λ0
    return BlockNLPModel(blocks, coeff, λ)
end

