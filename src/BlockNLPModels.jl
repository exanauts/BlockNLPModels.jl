module BlockNLPModels

using NLPModels
using LinearAlgebra
using SparseArrays

export AbstractBlockNLPModel
export BlockNLPModel
export DualizedNLPblockModel
export FullSpaceModel
export add_block
export add_links

abstract type AbstractBlockNLPModel end
abstract type AbstractLinkConstraint end

# To Do: Either sort the dictionaries or use a different representation

# To keep a count of different objects attached to the model
mutable struct BlockNLP_Counters
  block_counter::Int
  link_counter::Int
  var_counter::Int
  function BlockNLP_Counters()
      return new(0,0,0)
  end    
end

mutable struct AbstractBlockModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  problem_block::AbstractNLPModel
  idx::Int
  var_idx::UnitRange{Int} # Indices of variables wrt the full space model
  linking_constraints::Vector{Int} # ID of linking constraints that connect this block with other blocks 
end

mutable struct LinkConstraint <: AbstractLinkConstraint
  linking_matrices::Vector{SparseMatrixCSC{Float64,Int}} # linking_matrices[i] will give out the corresponsing A_i
  link_map::Dict{Int, Vector{Int}} # constraint => blocks, i.e., which constraint connects which blocks
  RHS_vector::SparseVector{Float64, Int}
  function LinkConstraint()
      return new(
          Vector{SparseMatrixCSC{Float64,Int}}(), 
          Dict{Int, Vector{Int}}(), 
          SparseVector{Float64, Int}(0, Vector{Int}(), Vector{Float64}())
          )
  end 
end

mutable struct BlockNLPModel <: AbstractBlockNLPModel
  problem_size::BlockNLP_Counters
  blocks::Vector{AbstractBlockModel}
  linking_constraints::AbstractLinkConstraint
  function BlockNLPModel()
    return new(
      BlockNLP_Counters(),
      Vector{AbstractBlockModel}(),
      LinkConstraint()
    )
  end
end

function add_block(block_nlp::AbstractBlockNLPModel, nlp::AbstractNLPModel)
  block_nlp.problem_size.block_counter += 1
  temp = deepcopy(block_nlp.problem_size.var_counter)+1
  block_nlp.problem_size.var_counter += nlp.meta.nvar
  var_idx = temp:block_nlp.problem_size.var_counter
  meta = nlp.meta
  push!(block_nlp.blocks, AbstractBlockModel(meta, Counters(), 
  nlp, block_nlp.problem_size.block_counter, var_idx, Vector{Int}()))
end

# This function must be called after blocks have been added
function add_links(block_nlp::AbstractBlockNLPModel, n_constraints::Int, 
  links::Dict{Int, SparseMatrixCSC{Float64, Int64}}, constants::SparseVector{Float64, Int})
  # Add n_constraints rows to the linking matrices
  for i in 1:block_nlp.problem_size.block_counter
      if block_nlp.problem_size.link_counter == 0
          push!(block_nlp.linking_constraints.linking_matrices, 
          spzeros(n_constraints, block_nlp.blocks[i].meta.nvar))
      else
          block_nlp.linking_constraints.linking_matrices[i] = 
          vcat(block_nlp.linking_constraints.linking_matrices[i], 
          spzeros(n_constraints, block_nlp.blocks[i].meta.nvar))
      end
  end
  # Add n_constraint rows to the RHS constant vector
  if block_nlp.problem_size.link_counter == 0
      block_nlp.linking_constraints.RHS_vector = constants
  else
      block_nlp.linking_constraints.RHS_vector = 
      vcat(block_nlp.linking_constraints.RHS_vector, constants)
  end

  # check if all dictionary elements are of appropriate sizes
  for block_idx in keys(links)
      @assert size(links[block_idx])[1] == n_constraints
      @assert size(links[block_idx])[2] == block_nlp.blocks[block_idx].meta.nvar
  end

  # Parse the constraints one-by-one
  for con in 1:n_constraints
      block_nlp.problem_size.link_counter += 1
      block_nlp.linking_constraints.link_map[block_nlp.problem_size.link_counter] = Vector{Int}()
      for block_idx in keys(links)
          if links[block_idx][con, :] != spzeros(block_nlp.blocks[block_idx].meta.nvar)
              push!(block_nlp.blocks[block_idx].linking_constraints, block_nlp.problem_size.link_counter)
              block_nlp.linking_constraints.linking_matrices[block_idx][block_nlp.problem_size.link_counter, :] = 
              links[block_idx][con, :]
              push!(block_nlp.linking_constraints.link_map[block_nlp.problem_size.link_counter], block_idx)
          end
      end
  end
end

include("dualized_block.jl")
include("full_space.jl")
end # module