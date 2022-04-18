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
        ncon = sum(m.blocks[i].meta.ncon for i in 1:nb) + 
        m.problem_size.link_counter,
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

"""
    get_hessian(
        m::FullSpaceModel; 
        x::Union{AbstractVector, Nothing} = nothing, 
        obj_weight::Union{AbstractVector, Nothing} = nothing, 
        y::Union{AbstractVector, Nothing} = nothing
    )
Returns the Hessian for a `FullSpaceModel` as a sparse matrix.
This function fills the nonzero entries of the Hessian matrix with ones if an `x` is not specified.

# Arguments
- `m::FullSpaceModel`: the full space model 
- `x::Union{AbstractVector, Nothing}`: current primal solution (optional)
- `obj_weight::Union{AbstractVector, Nothing}`: objective weight
- `y::Union{AbstractVector, Nothing}`: vector of dual variables
"""
function get_hessian(
    m::FullSpaceModel; 
    x::Union{AbstractVector, Nothing} = nothing, 
    obj_weight::Union{Number, Nothing} = nothing, 
    y::Union{AbstractVector, Nothing} = nothing
    )

    nb = m.blocknlp.problem_size.block_counter
    H = spzeros(m.meta.nvar, m.meta.nvar)

    if x === nothing
        for i in 1:nb
            block = m.blocknlp.blocks[i]

            H[block.var_idx, block.var_idx] = 
            sparse(
                hess_structure(block.problem_block)[1], 
                hess_structure(block.problem_block)[2], 
                ones(block.meta.nnzh), 
                block.meta.nvar, 
                block.meta.nvar
            )
        end
    else
        for i in 1:nb
            block = m.blocknlp.blocks[i]

            if block.meta.ncon > 0
                H[block.var_idx, block.var_idx] = 
                sparse(
                    hess_structure(block.problem_block)[1], 
                    hess_structure(block.problem_block)[2], 
                    hess_coord(
                        block.problem_block, 
                        x[block.var_idx], 
                        y[block.con_idx], 
                        obj_weight = obj_weight
                    ), 
                    block.meta.nvar, 
                    block.meta.nvar
                )
            else
                H[block.var_idx, block.var_idx] = 
                sparse(
                    hess_structure(block.problem_block)[1], 
                    hess_structure(block.problem_block)[2], 
                    hess_coord(
                        block.problem_block, 
                        x[block.var_idx], 
                        obj_weight = obj_weight
                    ), 
                    block.meta.nvar, 
                    block.meta.nvar
                )
            end        
        end
    end
    return H
end

"""
    get_jacobian(
        m::FullSpaceModel; 
        x::Union{AbstractVector, Nothing} = nothing 
    )
Returns the Jacobian for a `FullSpaceModel` as a sparse matrix.
This function fills the nonzero entries of the Hessian matrix with ones if an `x` is not specified.

# Arguments
- `m::FullSpaceModel`: the full space model 
- `x::Union{AbstractVector, Nothing}`: current primal solution (optional)
"""
function get_jacobian(m::FullSpaceModel; x::Union{AbstractVector, Nothing} = nothing)
    nb = m.blocknlp.problem_size.block_counter
    J = spzeros(n_constraints(m.blocknlp), m.blocknlp.problem_size.var_counter)
    if x === nothing
        for i in 1:nb
            block = m.blocknlp.blocks[i]
            if block.meta.ncon > 0
                J[block.con_idx, block.var_idx] = 
                sparse(
                    jac_structure(block.problem_block)[1], 
                    jac_structure(block.problem_block)[2], 
                    ones(block.meta.nnzj), 
                    block.meta.ncon, 
                    block.meta.nvar
                )
            end
        end
    else
        for i in 1:nb
            block = m.blocknlp.blocks[i]
            if block.meta.ncon > 0
                J[block.con_idx, block.var_idx] = 
                sparse(
                    jac_structure(block.problem_block)[1], 
                    jac_structure(block.problem_block)[2], 
                    jac_coord(block.problem_block, x[block.var_idx]), 
                    block.meta.ncon, 
                    block.meta.nvar
                )
            end
        end
    end
    J[m.blocknlp.problem_size.con_counter+1:end, :] = get_linking_matrix(m.blocknlp)
    
    return J
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
        gx[nlp.blocknlp.blocks[i].var_idx] = 
        grad(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[i].var_idx])
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
    H = get_hessian(nlp)
    rows, cols, dummy_vals = findnz(H)
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
    H = get_hessian(nlp, x = x, obj_weight = obj_weight)
    rows, cols, vals = findnz(H)
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
    H = get_hessian(nlp, x = x, y = y, obj_weight = obj_weight)
    rows, cols, vals = findnz(H)
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
            temp = idx+1
            idx += nlp.blocknlp.blocks[i].meta.ncon
            cx[temp:idx] = cons(nlp.blocknlp.blocks[i].problem_block, 
            x[nlp.blocknlp.blocks[i].var_idx])
        end
    end
    idx += 1
    A = get_linking_matrix_blocks(nlp.blocknlp)
    cx[idx:end] = sum(A[j]*x[nlp.blocknlp.blocks[j].var_idx] for j = 1:nb)
    return cx
end

function NLPModels.jac_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, 
cols::AbstractVector{T}) where {T}
    @lencheck nlp.meta.nnzj rows cols
    J = get_jacobian(nlp)
    rows, cols, vals = findnz(J)
    return rows, cols
end

function NLPModels.jac_coord!(nlp::FullSpaceModel, x::AbstractVector, vals::AbstractVector)
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nnzj vals 
    increment!(nlp, :neval_jac)
    J = get_jacobian(nlp, x = x)
    rows, cols, vals = findnz(J)
    return vals
end
  
  