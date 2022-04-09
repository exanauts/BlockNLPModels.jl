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

"""
    AbstractBlockNLPModel
Abstract supertype for the definition of NLP problems with a block structure.
"""
abstract type AbstractBlockNLPModel end

"""
    AbstractLinearLinkConstraint
Abstract supertype for the definition of linear linking constraints
"""
abstract type AbstractLinearLinkConstraint end

"""
    AbstractNLLinkConstraint
Abstract supertype for the definition of nonlinear linking constraints
"""
abstract type AbstractNLLinkConstraint end

include("utils.jl")

"""
  BlockNLP_Counters()
Keeps a count of blocks, linking constraints, variables, and (block) constraints attached to the model.
"""
mutable struct BlockNLP_Counters
  block_counter::Int
  link_counter::Int
  var_counter::Int
  con_counter::Int
  function BlockNLP_Counters()
      return new(0,0,0,0)
  end    
end

"""
    AbstractBlockModel{T, S} <: AbstractNLPModel{T, S}    
A data type to store block subproblems of the form
```math
\\begin{aligned}
  \\min_{x \\in \\mathbb{R^{m_i}}} \\quad & f_i(x_i) \\\\
  \\mathrm{subject \\, to} \\quad & c_{ij} (x_i) \\leq 0 \\quad \\forall j \\in \\mathcal{C}_i, \\\\
\\end{aligned}
where ``i`` is the index of the block.
"""
mutable struct AbstractBlockModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  problem_block::AbstractNLPModel
  idx::Int
  var_idx::UnitRange{Int} # Indices of variables wrt the full space model
  con_idx::UnitRange{Int} # Indices of constraints wrt the full space model
  linking_constraint_id::Vector{Int} # ID of linking constraints that connect this block with other blocks 
end


"""
    LinearLinkConstraint <: AbstractLinearLinkConstraint
A data type to store linear link constraints of the form
```math
\\begin{aligned}
  \\sum\\limits_{i \\in \\mathcal{B}} A_i x_i = b
\\end{aligned}
"""
mutable struct LinearLinkConstraint <: AbstractLinearLinkConstraint
  linking_blocks::Vector{SparseMatrixCSC{Float64,Int}} # linking_matrices[i] will give out the corresponsing A_i
  idx::UnitRange{Int}
  link_map::Dict{Int, Vector{Int}} # constraint => blocks, i.e., which constraint connects which blocks
  RHS_vector::Vector{Float64}
end

mutable struct NLLinkConstraint <: AbstractNLLinkConstraint
end

"""
    BlockNLPModel <: AbstractBlockNLPModel
A data type designed to store nonlinear optimization models of the form
```math
\\begin{aligned}
  \\min_{x \\in \\mathbb{R^{m_i}}} \\quad & \\sum\\limits_{i \\in \\mathcal{B}} f_i(x_i) \\\\
  \\mathrm{subject \\, to} \\quad & c_{ij} (x_i) \\leq 0 \\quad \\forall i \\in \\mathcal{B}, j \\in \\mathcal{C}_i \\\\
  & \\sum\\limits_{i \\in \\mathcal{B}} A_i x_i = b,
\\end{aligned}
where ``\\mathcal{B}`` is the set of variable blocks.

---

    BlockNLPModel()

Initializes an empty `BlockNLPModel`. 
"""
mutable struct BlockNLPModel <: AbstractBlockNLPModel
  problem_size::BlockNLP_Counters
  blocks::Vector{AbstractBlockModel}
  linking_constraints::Vector{Union{AbstractLinearLinkConstraint, AbstractNLLinkConstraint}}
  function BlockNLPModel()
    return new(
      BlockNLP_Counters(),
      Vector{AbstractBlockModel}(),
      Vector{Union{AbstractLinearLinkConstraint, AbstractNLLinkConstraint}}()
    )
  end
end

"""
  add_block(
    block_nlp::AbstractBlockNLPModel, 
    nlp::AbstractNLPModel
  )

Adds a block subproblem to a `BlockNLPModel`.
The `BlockNLPModel` needs to be initialized before this function can be called.

# Arguments

- `block_nlp::AbstractBlockNLPModel`: name of the BlockNLPModel to which a subproblem is to be added.
- `nlp::AbstractNLPModel`: the subproblem.
"""
function add_block(block_nlp::AbstractBlockNLPModel, nlp::AbstractNLPModel)
  block_nlp.problem_size.block_counter += 1

  temp = deepcopy(block_nlp.problem_size.var_counter)+1
  block_nlp.problem_size.var_counter += nlp.meta.nvar
  var_idx = temp:block_nlp.problem_size.var_counter

  temp = deepcopy(block_nlp.problem_size.con_counter)+1 
  block_nlp.problem_size.con_counter += nlp.meta.ncon
  con_idx = temp:block_nlp.problem_size.con_counter
  push!(block_nlp.blocks, AbstractBlockModel(nlp.meta, Counters(), 
  nlp, block_nlp.problem_size.block_counter, var_idx, con_idx, Vector{Int}()))
end

"""
    add_links(block_nlp::AbstractBlockNLPModel, 
    n_constraints::Int,   
    links::Union{Dict{Int, Array{Float64, 2}}, Dict{Int, Vector{Float64}}}, 
    constants::Union{AbstractVector, Float64})

Creates linear link constraint(s) between different subproblems of the BlockNLPModel.
This function must be called after subproblems have already been added.

# Arguments

- `block_nlp::AbstractBlockNLPModel`: The BlockNLPModel to which the constraint(s) is to be added.  
- `n_constraints::Int`: Number of link constraints. 
- `links::Union{Dict{Int, Array{Float64, 2}}, Dict{Int, Vector{Float64}}}`: coefficients of the block matrices that link different blocks,
- `constants::Union{AbstractVector, Float64}`: RHS vector
"""
# 
function add_links(block_nlp::AbstractBlockNLPModel, n_constraints::Int, 
  links::Union{Dict{Int, Array{Float64, 2}}, Dict{Int, Vector{Float64}}}, 
  constants::Union{AbstractVector, Float64})
  
  linking_blocks = Vector{SparseMatrixCSC{Float64, Int}}(undef, block_nlp.problem_size.block_counter)
  RHS_vector = Vector{Float64}(undef, n_constraints)

  # Initialize linking_blocks with empty matrices
  for i in 1:block_nlp.problem_size.block_counter
    linking_blocks[i] = spzeros(n_constraints, block_nlp.blocks[i].meta.nvar)
  end
  
  # check if everything is of appropriate size and add to linking_blocks
  @assert length(constants) == n_constraints
  if n_constraints == 1
    RHS_vector[1] = constants
    for block_idx in keys(links)
      @assert length(links[block_idx]) == block_nlp.blocks[block_idx].meta.nvar
      linking_blocks[block_idx][1, :] = sparse(links[block_idx])  
    end
  else
    RHS_vector = constants
    for block_idx in keys(links)
      @assert size(links[block_idx])[1] == n_constraints
      @assert size(links[block_idx])[2] == block_nlp.blocks[block_idx].meta.nvar
      linking_blocks[block_idx] = sparse(links[block_idx])
    end
  end
  
  # Prepare other information
  link_map = Dict{Int, Vector{Int}}()
  link_con_idx = Vector{Int}()

  # Parse the constraints one-by-one to collect this information
  temp = block_nlp.problem_size.link_counter+1
  for con in 1:n_constraints
      block_nlp.problem_size.link_counter += 1
      link_map[block_nlp.problem_size.link_counter] = Vector{Int}()
      for block_idx in keys(links)
          if linking_blocks[block_idx][con, :] != spzeros(block_nlp.blocks[block_idx].meta.nvar)
              push!(block_nlp.blocks[block_idx].linking_constraint_id, block_nlp.problem_size.link_counter)
              push!(link_map[block_nlp.problem_size.link_counter], block_idx)
          end
      end
  end
  link_con_idx = temp:block_nlp.problem_size.link_counter
  lin_link_con = LinearLinkConstraint(linking_blocks, link_con_idx, link_map, RHS_vector)
  push!(block_nlp.linking_constraints, lin_link_con)
end

include("dualized_block.jl")
include("full_space.jl")
end # module