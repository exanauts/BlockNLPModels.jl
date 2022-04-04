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
          nnzh = sum((m.blocks[i].meta.nvar)^2 for i in 1:nb),
          nnzj = sum(m.blocks[i].meta.ncon*m.blocks[i].meta.nvar for i in 1:nb) + 
          n_var*m.problem_size.link_counter,
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
      return sum(obj(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[i].var_idx]) for i in 1:nb)
  end
  
  function NLPModels.grad!(nlp::FullSpaceModel, x::AbstractVector, gx::AbstractVector)
      nb = nlp.blocknlp.problem_size.block_counter
      @lencheck nlp.meta.nvar x gx
      increment!(nlp, :neval_grad)
      for i in 1:nb
          gx[nlp.blocknlp.blocks[i].var_idx] = grad(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[i].var_idx])
      end
      return gx
  end
  
  function NLPModels.hess_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
      nb = nlp.blocknlp.problem_size.block_counter
      @lencheck nlp.meta.nnzh rows cols
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
      nb = nlp.blocknlp.problem_size.block_counter
      @lencheck nlp.meta.nvar x
      @lencheck nlp.meta.ncon cx
      increment!(nlp, :neval_cons)
      idx = 0
      for i in 1:nb
          if nlp.blocknlp.blocks[i].meta.ncon > 0
              temp = deepcopy(idx)+1
              idx += nlp.blocknlp.blocks[i].meta.ncon
              cx[temp:idx] = cons(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[i].var_idx])
          end
      end
      idx += 1
      cx[idx:end] = sum(nlp.blocknlp.linking_constraints.linking_matrices[j]*x[nlp.blocknlp.blocks[j].var_idx] for j = 1:nb)
      return cx
  end
  
  function NLPModels.jac_structure!(nlp::FullSpaceModel, rows::AbstractVector{T}, cols::AbstractVector{T}) where {T}
      nb = nlp.blocknlp.problem_size.block_counter
      @lencheck nlp.meta.nnzj rows cols
      row_idx = 0
      idx = 0
      for i in 1:nb
          if nlp.blocknlp.blocks[i].meta.ncon > 0
              for j in 1:nlp.blocknlp.blocks[i].meta.ncon
                  row_idx += 1
                  temp = deepcopy(idx)+1
                  idx += nlp.blocknlp.blocks[i].meta.nvar
                  rows[temp:idx] .= row_idx
                  cols[temp:idx] = collect(nlp.blocknlp.blocks[i].var_idx)
              end
          end
      end
      for i in 1:nlp.blocknlp.problem_size.link_counter
          row_idx += 1
          temp = deepcopy(idx)+1
          idx += nlp.meta.nvar
          rows[temp:idx] .= row_idx
          cols[temp:idx] = collect(1:nlp.meta.nvar)
      end
      return rows, cols
  end
  
  function NLPModels.jac_coord!(nlp::FullSpaceModel, x::AbstractVector, vals::AbstractVector)
      nb = nlp.blocknlp.problem_size.block_counter
      @lencheck nlp.meta.nvar x
      @lencheck nlp.meta.nnzj vals 
      increment!(nlp, :neval_jac)
      idx = 0
      for i in 1:nb
          if nlp.blocknlp.blocks[i].meta.ncon > 0
              temp = deepcopy(idx)+1
              idx += nlp.blocknlp.blocks[i].meta.nvar * nlp.blocknlp.blocks[i].meta.ncon
              vals[temp:idx] = jac_coord(nlp.blocknlp.blocks[i].problem_block, x[nlp.blocknlp.blocks[i].var_idx])
          end
      end
      temp_A = nlp.blocknlp.linking_constraints.linking_matrices[1]
      for i in 2:nb
          temp_A = hcat(temp_A, nlp.blocknlp.linking_constraints.linking_matrices[i])
      end
      for j in 1:nlp.blocknlp.problem_size.link_counter
          for k in 1:nlp.meta.nvar
              idx += 1
              vals[idx] = temp_A[j, k]
          end
      end
      return vals
  end
  
  