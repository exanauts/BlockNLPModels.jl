"""
    FullSpaceModel{T, S} <: AbstractNLPModel{T, S}   

A data type to store the full space `BlockNLPModel` as an `AbstractNLPModel`.
"""
mutable struct FullSpaceModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    blocknlp::AbstractBlockNLPModel
  end

"""
FullSpaceModel(
    ::Type{T}, m::AbstractBlockNLPModel
    ) where {T}

Converts a `BlockNLPModel` to a `AbstractNLPModel` that can be solved with any standard NLP solver.

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel
"""
function FullSpaceModel(m::AbstractBlockNLPModel)
    nb = m.problem_size.block_counter
    n_var = sum(m.blocks[i].meta.nvar for i in 1:nb)

    l_con::Vector{Float64} = []
    u_con::Vector{Float64} = []
    for i in 1:nb
        if m.blocks[i].meta.ncon > 0
            l_con = vcat(l_con, m.blocks[i].meta.lcon)
            u_con = vcat(u_con, m.blocks[i].meta.ucon)
        end
    end
    l_con = vcat(l_con, get_rhs_vector(m))
    u_con = vcat(u_con, get_rhs_vector(m))

    l_var::Vector{Float64} = []
    u_var::Vector{Float64} = []
    for i in 1:nb
        l_var = vcat(l_var, m.blocks[i].meta.lvar)
        u_var = vcat(u_var, m.blocks[i].meta.uvar)
    end
    meta = NLPModelMeta(
        n_var,
        ncon = n_constraints(m),
        nnzh = sum(m.blocks[i].meta.nnzh for i in 1:nb),
        nnzj = sum(m.blocks[i].meta.nnzj for i in 1:nb) + 
        nnz(get_linking_matrix(m)),
        x0 = zeros(Float64, n_var),
        lvar = l_var,
        uvar = u_var,
        lcon = l_con,
        ucon = u_con,
        minimize = true,
        name = "full_space",
    )
    return FullSpaceModel(meta, Counters(), m)
end

function NLPModels.obj(nlp::FullSpaceModel, x::AbstractVector)
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_obj)
    return sum(obj(nlp.blocknlp.blocks[i].problem_block, 
    x[nlp.blocknlp.blocks[i].var_idx]) for i in 1:nb)
end
  
function NLPModels.grad!(nlp::FullSpaceModel, x::AbstractVector, gx::AbstractVector)
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nvar x gx
    increment!(nlp, :neval_grad)
    for i in 1:nb
        v = @view gx[nlp.blocknlp.blocks[i].var_idx]
        grad!(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[i].var_idx], v)
    end
    return gx
end

function NLPModels.hess_structure!(
    nlp::FullSpaceModel, 
    rows::AbstractVector{T}, 
    cols::AbstractVector{T}
    ) where {T}
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nnzh rows cols
    idx = 0
    for i in 1:nb
        block = nlp.blocknlp.blocks[i]
        temp_idx = idx+1
        idx += block.meta.nnzh
        v1 = @view rows[temp_idx:idx]
        v2 = @view cols[temp_idx:idx]
        hess_structure!(block.problem_block, v1, v2)
        v1 .+= (block.var_idx[1] - 1)
        v2 .+= (block.var_idx[1] - 1)
    end
    return rows, cols
end

function NLPModels.hess_coord!(
nlp::FullSpaceModel,
x::AbstractVector{T},
vals::AbstractVector{T};
obj_weight = one(T),
) where {T}
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nnzh vals
    increment!(nlp, :neval_hess)
    idx = 0
    for i in 1:nb
        block = nlp.blocknlp.blocks[i]
        temp_idx = idx+1
        idx += block.meta.nnzh
        v = @view vals[temp_idx:idx]
        hess_coord!(block.problem_block, x[block.var_idx], v, obj_weight = obj_weight)
    end
    return vals
end

function NLPModels.hess_coord!(
    nlp::FullSpaceModel,
    x::AbstractVector{T},
    y::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight = one(T),
    ) where {T}
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nnzh vals
    @lencheck nlp.meta.ncon y
    increment!(nlp, :neval_hess)
    idx = 0
    for i in 1:nb
        block = nlp.blocknlp.blocks[i]
        if block.meta.ncon == 0
            temp_idx = idx+1
            idx += block.meta.nnzh
            v = @view vals[temp_idx:idx]
            hess_coord!(block.problem_block, x[block.var_idx], v, obj_weight = obj_weight)
        else
            temp_idx = idx+1
            idx += block.meta.nnzh
            v = @view vals[temp_idx:idx]
            hess_coord!(block.problem_block, x[block.var_idx], y[block.con_idx], v, obj_weight = obj_weight)
        end
    end
    return vals
end

function NLPModels.cons!(nlp::FullSpaceModel, x::AbstractVector, cx::AbstractVector)
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.ncon cx
    increment!(nlp, :neval_cons)
    idx = 0
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
            temp_idx = idx+1
            idx += nlp.blocknlp.blocks[i].meta.ncon
            v = @view cx[temp_idx:idx]
            cons!(nlp.blocknlp.blocks[i].problem_block, 
            x[nlp.blocknlp.blocks[i].var_idx], v)
        end
    end
    idx += 1
    A = get_linking_matrix_blocks(nlp.blocknlp)
    cx[idx:end] .= sum(A[j]*x[nlp.blocknlp.blocks[j].var_idx] for j = 1:nb)
    return cx
end

function NLPModels.jac_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, 
cols::AbstractVector{T}) where {T}
    @lencheck nlp.meta.nnzj rows cols
    nb = nlp.blocknlp.problem_size.block_counter
    idx = 0
    for i in 1:nb
        block = nlp.blocknlp.blocks[i]
        temp_idx = idx+1
        idx += block.meta.nnzj
        v1 = @view rows[temp_idx:idx]
        v2 = @view cols[temp_idx:idx]
        jac_structure!(block.problem_block, v1, v2)
        v1 .+= (block.con_idx[1] - 1)
        v2 .+= (block.var_idx[1] - 1)
    end
    r, c, v = findnz(get_linking_matrix(nlp.blocknlp))
    rows[idx+1:end] .= r .+ nlp.blocknlp.problem_size.con_counter
    cols[idx+1:end] .= c
    return rows, cols
end

function NLPModels.jac_coord!(nlp::FullSpaceModel, x::AbstractVector, vals::AbstractVector)
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nnzj vals 
    increment!(nlp, :neval_jac)
    nb = nlp.blocknlp.problem_size.block_counter
    idx = 0
    for i in 1:nb
        block = nlp.blocknlp.blocks[i]
        temp_idx = idx+1
        idx += block.meta.nnzj
        v1 = @view vals[temp_idx:idx]
        jac_coord!(block.problem_block, x[block.var_idx], v1)
    end
    r, c, v = findnz(get_linking_matrix(nlp.blocknlp))
    vals[idx+1:end] .= v
    return vals
end
  
  