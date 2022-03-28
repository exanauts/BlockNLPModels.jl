using BlockNLPModels
using NLPModels

mutable struct FullSpaceModel{T, S} <: AbstractNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  blocknlp::AbstractBlockNLPModel
end

function FullSpaceModel(::Type{T}, m::AbstractBlockNLPModel) where {T}
    nb = length(m.blocks)
    l_con::Vector{Float64} = []
    u_con::Vector{Float64} = []
    for i in 1:nb
        if m.blocks[i].meta.ncon > 0
            push!(l_con, m.blocks[i].meta.lvar[1])
            push!(u_con, m.blocks[i].meta.uvar[1])
        end
    end
    push!(l_con, m.linkconstraints[nb+1])
    push!(u_con, m.linkconstraints[nb+1])
    meta = NLPModelMeta{T, Vector{T}}(
        nb,
        ncon = sum(m.blocks[i].meta.ncon for i in 1:nb) + 1,
        nnzh = nb,
        nnzj = sum(m.blocks[i].meta.ncon for i in 1:nb) + nb, # ncon - 1 + nb
        x0 = zeros(Float64, nb),
        lvar = [m.blocks[i].meta.lvar[1] for i in 1:nb],
        uvar = [m.blocks[i].meta.uvar[1] for i in 1:nb],
        lcon = l_con,
        ucon = u_con,
        minimize = true,
        name = "full_space",
    )
  return FullSpaceModel(meta, Counters(), m)
end

FullSpaceModel(m::AbstractBlockNLPModel) = FullSpaceModel(Float64, m)

function NLPModels.obj(nlp::FullSpaceModel, x::AbstractVector)
    nb = length(nlp.blocknlp.blocks)
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_obj)
    return sum(obj(nlp.blocknlp.blocks[i], [x[i]]) for i in 1:nb)
end

function NLPModels.grad!(nlp::FullSpaceModel, x::AbstractVector, gx::AbstractVector)
    nb = length(nlp.blocknlp.blocks)
    @lencheck nlp.meta.nvar x gx
    increment!(nlp, :neval_grad)
    for i in 1:nb
        gx[i] = grad(nlp.blocknlp.blocks[i], [x[i]])[1]
    end
    return gx
end

function NLPModels.hess_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    nb = length(nlp.blocknlp.blocks)
    @lencheck nb rows cols
    for i in 1:nb
        rows[i] = i
        cols[i] = i
    end
    return rows, cols
end

function NLPModels.hess_coord!(
  nlp::FullSpaceModel,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight = one(T),
) where {T}
    nb = length(nlp.blocknlp.blocks)
    @lencheck nb x vals
    increment!(nlp, :neval_hess)
    for i in 1:nb
        vals[i] = hess_coord(nlp.blocknlp.blocks[i], [x[i]], obj_weight = obj_weight)[1]
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
    nb = length(nlp.blocknlp.blocks)
    @lencheck nb x vals
    @lencheck nlp.meta.ncon y
    increment!(nlp, :neval_hess)
    idx = 0
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
            idx += 1
            vals[i] = hess_coord(nlp.blocknlp.blocks[i], [x[i]], [y[i]], obj_weight = obj_weight)[1]
        else
            idx += 1
            vals[i] = hess_coord(nlp.blocknlp.blocks[i], [x[i]], obj_weight = obj_weight)[1]
        end
    end
    return vals
end

function NLPModels.cons!(nlp::FullSpaceModel, x::AbstractVector, cx::AbstractVector)
    nb = length(nlp.blocknlp.blocks)
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.ncon cx
    increment!(nlp, :neval_cons)
    idx = 0
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
            idx += 1
            cx[idx] = cons(nlp.blocknlp.blocks[i], [x[i]])[1]
        end
    end
    cx[end] = sum(nlp.blocknlp.linkconstraints[j]*x[j] for j = 1:nb)
    return cx
end

function NLPModels.jac_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
    nb = length(nlp.blocknlp.blocks)
    @lencheck (nlp.meta.ncon - 1 + nb) rows cols
    idx = 0
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
            idx += 1
            rows[idx] = idx
            cols[idx] = idx
        end
    end
    for i in 1:nb
        rows[idx+i] = idx+1
        cols[idx+i] = idx+i
    end
    return rows, cols
end

function NLPModels.jac_coord!(nlp::FullSpaceModel, x::AbstractVector, vals::AbstractVector)
    nb = length(nlp.blocknlp.blocks)
    @lencheck nb x
    @lencheck (nlp.meta.ncon - 1 + nb)  vals 
    increment!(nlp, :neval_jac)
    idx = 0
    for i in 1:nb
        if nlp.blocknlp.blocks[i].meta.ncon > 0
            idx += 1
            vals[idx] = jac_coord(nlp.blocknlp.blocks[i], [x[i]])[1]
        end
    end
    for i in 1:nb
        vals[idx+i] = nlp.blocknlp.linkconstraints[i]
    end
    return vals
end

