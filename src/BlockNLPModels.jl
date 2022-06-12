module BlockNLPModels

using NLPModels
using LinearAlgebra
using SparseArrays

export AbstractBlockNLPModel
export BlockNLPModel
export DualizedNLPBlockModel
export AugmentedNLPBlockModel
export FullSpaceModel
export add_block, add_links, n_constraints,
       get_linking_matrix, get_linking_matrix_blocks, 
       get_rhs_vector, update_dual!, update_primal!

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
    AbstractNonLinearLinkConstraint
Abstract supertype for the definition of nonlinear linking constraints
"""
abstract type AbstractNonLinearLinkConstraint end

include("utils.jl")

"""
    BlockNLPCounters()
Keeps a count of blocks, linking constraints, variables, and (block) constraints attached to the model.
"""
mutable struct BlockNLPCounters
    block_counter::Int
    link_counter::Int
    var_counter::Int
    con_counter::Int
    function BlockNLPCounters()
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
    linking_blocks::Vector{SparseMatrixCSC{Float64,Int}}
    idx::UnitRange{Int}
    link_map::Dict{Int, Vector{Int}} # constraint => blocks, i.e., which constraint connects which blocks
    rhs_vector::Vector{Float64}
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
    problem_size::BlockNLPCounters
    blocks::Vector{AbstractBlockModel}
    linking_constraints::Vector{Union{AbstractLinearLinkConstraint, AbstractNonLinearLinkConstraint}}
    function BlockNLPModel()
        return new(
                   BlockNLPCounters(),
                   Vector{AbstractBlockModel}(),
                   Vector{Union{AbstractLinearLinkConstraint, AbstractNonLinearLinkConstraint}}()
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

    temp_var_counter = block_nlp.problem_size.var_counter + 1
    block_nlp.problem_size.var_counter += nlp.meta.nvar
    var_idx = temp_var_counter:block_nlp.problem_size.var_counter

    temp_con_counter = block_nlp.problem_size.con_counter + 1
    block_nlp.problem_size.con_counter += nlp.meta.ncon
    con_idx = temp_con_counter:block_nlp.problem_size.con_counter

    push!(block_nlp.blocks, AbstractBlockModel(nlp.meta, Counters(),
                                               nlp, block_nlp.problem_size.block_counter, var_idx, con_idx, Vector{Int}()))
end

"""
    add_links(block_nlp::AbstractBlockNLPModel, n_constraints::Int,
        links::Dict{Int, M},
        constants::Union{AbstractVector, Float64},
    ) where M <: AbstractMatrix{Float64}

Creates linear link constraint(s) between different subproblems of the BlockNLPModel.
This function must be called after subproblems have already been added.

# Arguments

- `block_nlp::AbstractBlockNLPModel`: The BlockNLPModel to which the constraint(s) is to be added.
- `n_constraints::Int`: Number of link constraints.
- `links::Union{Dict{Int, Array{Float64, 2}}, Dict{Int, Vector{Float64}}}`: coefficients of the block matrices that link different blocks,
- `constants::Union{AbstractVector, Float64}`: RHS vector
"""
function add_links(block_nlp::AbstractBlockNLPModel, n_constraints::Int,
    links::Dict{Int, M},
    constants::Union{AbstractVector, Float64},
) where M <: AbstractMatrix{Float64}

    # Initialize linking_blocks with empty matrices
    linking_blocks = [spzeros(n_constraints, block_nlp.blocks[i].meta.nvar)
                      for i in 1:block_nlp.problem_size.block_counter]

    # check if everything is of appropriate size and add to linking_blocks
    @assert length(constants) == n_constraints
    rhs_vector = constants
    for block_idx in keys(links)
        @assert size(links[block_idx])[1] == n_constraints
        @assert size(links[block_idx])[2] == block_nlp.blocks[block_idx].meta.nvar
        linking_blocks[block_idx] = sparse(links[block_idx])
    end

    # Prepare other information
    link_map = Dict{Int, Vector{Int}}()
    link_con_idx = Vector{Int}()

    # Parse the constraints one-by-one to collect this information
    temp_link_counter = block_nlp.problem_size.link_counter + 1
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
    link_con_idx = temp_link_counter:block_nlp.problem_size.link_counter
    lin_link_con = LinearLinkConstraint(linking_blocks, link_con_idx, link_map, rhs_vector)
    push!(block_nlp.linking_constraints, lin_link_con)
end

# Special treatment for single constraint case 
function add_links(
    block_nlp::AbstractBlockNLPModel, n_constraints::Int,
    links::Dict{Int, M},
    constants::Float64,
    ) where M <: AbstractVector{Float64}
    add_links(block_nlp,n_constraints,Dict(Matrix(link') for link in links),constants)
end

include("full_space.jl")
include("dualized_block.jl")
include("augmented_block.jl")

end # module
