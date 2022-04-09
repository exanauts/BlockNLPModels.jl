mutable struct FullSpaceModel{T, S} <: AbstractNLPModel{T, S}
    meta::NLPModelMeta{T, S}
    counters::Counters
    blocknlp::AbstractBlockNLPModel
  end
  
function FullSpaceModel(::Type{T}, m::AbstractBlockNLPModel) where {T}
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
    l_con = vcat(l_con, m.linking_constraints.RHS_vector)
    u_con = vcat(u_con, m.linking_constraints.RHS_vector)

    l_var::Vector{Float64} = []
    u_var::Vector{Float64} = []
    for i in 1:nb
        l_var = vcat(l_var, m.blocks[i].meta.lvar)
        u_var = vcat(u_var, m.blocks[i].meta.uvar)
    end

    meta = NLPModelMeta{T, Vector{T}}(
        n_var,
        ncon = sum(m.blocks[i].meta.ncon for i in 1:nb) + 
        m.problem_size.link_counter,
        nnzh = sum(m.blocks[i].meta.nnzh for i in 1:nb),
        nnzj = sum(m.blocks[i].meta.nnzj for i in 1:nb) + 
        nnz(get_blockmatrix(m)),
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

FullSpaceModel(m::AbstractBlockNLPModel) = FullSpaceModel(Float64, m)

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

function NLPModels.hess_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, 
cols::AbstractVector{T}) where {T}
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nnzh rows cols
    H = spzeros(nlp.meta.nvar, nlp.meta.nvar)
    for i in 1:nb
    H[nlp.blocknlp.blocks[i].var_idx, nlp.blocknlp.blocks[i].var_idx] = 
    sparse(hess_structure(nlp.blocknlp.blocks[i].problem_block[1]), 
    hess_structure(nlp.blocknlp.blocks[i].problem_block[1]), 
    ones(nlp.blocknlp.blocks[i].meta.nnzh), 
    nlp.blocknlp.blocks[i].meta.nvar, nlp.blocknlp.blocks[i].meta.nvar)
    end
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
    for i in 1:nb
    H[nlp.blocknlp.blocks[i].var_idx, nlp.blocknlp.blocks[i].var_idx] = 
    sparse(hess_structure(nlp.blocknlp.blocks[i].problem_block)[1], 
    hess_structure(nlp.blocknlp.blocks[i].problem_block)[2], 
    hess_coord(nlp.blocknlp.blocks[i].problem_block, 
    x[nlp.blocknlp.blocks[i].var_idx], 
    obj_weight = obj_weight), 
    nlp.blocknlp.blocks[i].meta.nvar, 
    nlp.blocknlp.blocks[i].meta.nvar)
    end
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

    H = spzeros(nlp.meta.nvar, nlp.meta.nvar)
    for i in 1:nb
    if nlp.blocknlp.blocks[i].meta.ncon > 0
        H[nlp.blocknlp.blocks[i].var_idx, nlp.blocknlp.blocks[i].var_idx] = 
        sparse(hess_structure(nlp.blocknlp.blocks[i].problem_block)[1], 
        hess_structure(nlp.blocknlp.blocks[i].problem_block)[2], 
        hess_coord(nlp.blocknlp.blocks[i].problem_block, 
        x[nlp.blocknlp.blocks[i].var_idx], 
        y[nlp.blocknlp.blocks[i].con_idx], 
        obj_weight = obj_weight), 
        nlp.blocknlp.blocks[i].meta.nvar, 
        nlp.blocknlp.blocks[i].meta.nvar)
    else
        H[nlp.blocknlp.blocks[i].var_idx, nlp.blocknlp.blocks[i].var_idx] = 
        sparse(hess_structure(nlp.blocknlp.blocks[i].problem_block)[1], 
        hess_structure(nlp.blocknlp.blocks[i].problem_block)[2], 
        hess_coord(nlp.blocknlp.blocks[i].problem_block, 
        x[nlp.blocknlp.blocks[i].var_idx], 
        obj_weight = obj_weight), 
        nlp.blocknlp.blocks[i].meta.nvar, 
        nlp.blocknlp.blocks[i].meta.nvar)
    end
    end
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
            temp = deepcopy(idx)+1
            idx += nlp.blocknlp.blocks[i].meta.ncon
            cx[temp:idx] = cons(nlp.blocknlp.blocks[i].problem_block, 
            x[nlp.blocknlp.blocks[i].var_idx])
        end
    end
    idx += 1
    A = get_matrixblocks(nlp.blocknlp)
    cx[idx:end] = sum(A[j]*x[nlp.blocknlp.blocks[j].var_idx] for j = 1:nb)
    return cx
end

function NLPModels.jac_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, 
cols::AbstractVector{T}) where {T}
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nnzj rows cols
    J = spzeros(n_tot_con(nlp.blocknlp), nlp.blocknlp.problem_size.var_counter)
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
        J[nlp.blocknlp.blocks[i].con_idx, nlp.blocknlp.blocks[i].var_idx] = 
        sparse(jac_structure(nlp.blocknlp.blocks[i].problem_block)[1], 
        jac_structure(nlp.blocknlp.blocks[i].problem_block)[2], ones(nlp.blocknlp.blocks[i].nnzj), 
        nlp.blocknlp.blocks[i].meta.ncon, nlp.blocknlp.blocks[i].meta.nvar)
        end
    end
    J[nlp.blocknlp.problem_size.con_counter+1:end, :] = get_blockmatrix(nlp.blocknlp)
    rows, cols, vals = findnz(J)
    return rows, cols
end

function NLPModels.jac_coord!(nlp::FullSpaceModel, x::AbstractVector, vals::AbstractVector)
    nb = nlp.blocknlp.problem_size.block_counter
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.nnzj vals 
    increment!(nlp, :neval_jac)
    J = spzeros(n_tot_con(nlp.blocknlp), nlp.blocknlp.problem_size.var_counter)
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
        J[nlp.blocknlp.blocks[i].con_idx, nlp.blocknlp.blocks[i].var_idx] = 
        sparse(jac_structure(nlp.blocknlp.blocks[i].problem_block)[1], 
        jac_structure(nlp.blocknlp.blocks[i].problem_block)[2], 
        jac_coord(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[1].var_idx]), 
        nlp.blocknlp.blocks[i].meta.ncon, nlp.blocknlp.blocks[i].meta.nvar)
        end
    end
    J[nlp.blocknlp.problem_size.con_counter+1:end, :] = get_blockmatrix(nlp.blocknlp)
    rows, cols, vals = findnz(J)
    return vals
end
  
  