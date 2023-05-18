"""
    get_linking_matrix(
        m::AbstractBlockNLPModel,
    )

Returns the concatenated linking matrix ``A = [A_1, \\ldots, A_B]``

# Arguments

- `m::AbstractBlockNLPModel`: BlockNLPModel whose block matrix is to be extracted.
"""
function get_linking_matrix(m::AbstractBlockNLPModel)
    nb = m.problem_size.block_counter
    A = spzeros(m.problem_size.link_counter, m.problem_size.var_counter)
    for i in 1:nb
        for j in 1:length(m.linking_constraints)
            A[m.linking_constraints[j].idx, m.blocks[i].var_idx] =
                m.linking_constraints[j].linking_blocks[i]
        end
    end
    return A
end

"""
    get_linking_matrix_blocks(
        m::AbstractBlockNLPModel
    )

Returns a vector of linking matrix blocks ``[A_1, \\ldots, A_B]``

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel whose block matrices are to be extracted.
"""
function get_linking_matrix_blocks(m::AbstractBlockNLPModel)
    nb = m.problem_size.block_counter
    A = [spzeros(m.problem_size.link_counter, m.blocks[i].meta.nvar) for i = 1:nb]
    for i = 1:nb
        for j = 1:length(m.linking_constraints)
            A[i][m.linking_constraints[j].idx, :] =
                m.linking_constraints[j].linking_blocks[i]
        end
    end
    return A
end

"""
    get_rhs_vector(
        m::AbstractBlockNLPModel
    )

Returns the concatenated RHS vector for all the linking constraints ``b``.

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel whose RHS vector is to be extracted.
"""
function get_rhs_vector(m::AbstractBlockNLPModel)
    b = zeros(m.problem_size.link_counter)
    for j = 1:length(m.linking_constraints)
        b[m.linking_constraints[j].idx] = m.linking_constraints[j].rhs_vector
    end
    return b
end

"""
    n_constraints(
        m::AbstractBlockNLPModel
    )

Returns the total number of constraints in a BlockNLPModel.

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel.
"""
function n_constraints(m::AbstractBlockNLPModel)
    return m.problem_size.link_counter + m.problem_size.con_counter
end

"""
    update_nnzh(
        meta::NLPModelMeta,
        new_nnzh::Int
    )

Updates the nnzh field of NLPModelMeta.

# Arguments

- `meta::NLPModelMeta`: name of the BlockNLPModel.
- `new_nnzh`: new nnzh value to update the meta with.
"""
function update_nnzh(meta::NLPModelMeta, new_nnzh::Int)
    field_names = [
        :nvar,
        :x0,
        :lvar,
        :uvar,
        :nlvb,
        :nlvo,
        :nlvc,
        :ncon,
        :y0,
        :lcon,
        :ucon,
        :nnzo,
        :nnzj,
        :lin_nnzj,
        :nln_nnzj,
        :nnzh,
        :lin,
        :minimize,
        :islp,
        :name,
    ]
    field_values = [getfield(meta, field) for field in field_names]

    new_meta = Dict(zip(field_names, field_values))
    new_meta[:nnzh] = new_nnzh

    nvar = new_meta[:nvar]
    pop!(new_meta, :nvar)

    return NLPModels.NLPModelMeta(nvar; new_meta...)
end

"""
    AugmentedHessianInfo

A data type to store information required to compute augmented hessian for a NLP block.
"""
mutable struct AugmentedHessianInfo
    augmented_hessian_struct::Tuple{AbstractVector,AbstractVector}
    block_hessian_struct::Tuple{AbstractVector,AbstractVector}
    ATA::Tuple{AbstractVector,AbstractVector,AbstractVector}
end

"""
    get_augmented_hessian_coord!(nlp::AbstractNLPModel,
        x::AbstractVector,
        vals::AbstractVector,
        obj_weight::Number;
        y::Union{AbstractVector, Nothing} = nothing,
    )

Returns the Hessian for an augmented subproblem as a sparse matrix.

# Arguments
- `m::AbstractNLPModel`: the subproblem
- `x::Union{AbstractVector, Nothing}`: current primal solution (optional)
- `obj_weight::Union{AbstractVector, Nothing}`: objective weight
- `y::Union{AbstractVector, Nothing}`: vector of dual variables
- `vals::Union{AbstractVector, Nothing}`: nonzero values of the Hessian matrix
"""
function get_augmented_hessian_coord!(
    nlp::AbstractNLPModel,
    ρ::Number,
    x::AbstractVector,
    vals::AbstractVector,
    obj_weight::Number;
    y::Union{AbstractVector,Nothing} = nothing,
)
    # initialize vector indices
    # main_idx = 1
    # sub_idx1 = 1
    # sub_idx2 = 1

    # assign pointers
    # aug_hess = nlp.hess_info.augmented_hessian_struct
    blk_hess = nlp.hess_info.block_hessian_struct
    aug_term = nlp.hess_info.ATA
    if nlp.subproblem.meta.ncon > 0
        blk_hess_values =
            hess_coord(nlp.subproblem.problem_block, x, y, obj_weight = obj_weight)
    else
        blk_hess_values =
            hess_coord(nlp.subproblem.problem_block, x, obj_weight = obj_weight)
    end
    # This needs fixing
    I = vcat(blk_hess[1], aug_term[1])
    J = vcat(blk_hess[2], aug_term[2])
    V = vcat(blk_hess_values, ρ*obj_weight .* aug_term[3])
    aug_hess = findnz(sparse(I, J, V, nlp.meta.nvar, nlp.meta.nvar))
    vals .= aug_hess[3]
    # for (i, j) in zip(aug_hess[1], aug_hess[2])
    #     # println(sub_idx1)
    #     # println(sub_idx2)
    #     if (
    #         blk_hess1[sub_idx1] == i &&
    #         blk_hess2[sub_idx1] == j &&
    #         aug_term[1][sub_idx2] == i &&
    #         aug_term[2][sub_idx2] == j
    #     )
    #         vals[main_idx] = blk_hess_values[sub_idx1] + aug_term[3][sub_idx2]
    #         main_idx += 1
    #         (sub_idx1 += 1)
    #         (sub_idx2 += 1)
    #         if length(aug_term[3]) < sub_idx2
    #             sub_idx2 -= 1
    #         end
    #         if length(blk_hess_values) < sub_idx1
    #             sub_idx1 -= 1
    #         end
    #     elseif (blk_hess1[sub_idx1] == i && blk_hess2[sub_idx1] == j)
    #         vals[main_idx] = blk_hess_values[sub_idx1]
    #         main_idx += 1
    #         (sub_idx1 += 1)
    #         if length(blk_hess_values) < sub_idx1
    #             sub_idx1 -= 1
    #         end
    #     elseif (aug_term[1][sub_idx2] == i && aug_term[2][sub_idx2] == j)
    #         vals[main_idx] = aug_term[3][sub_idx2]
    #         main_idx += 1
    #         (sub_idx2 += 1)
    #         if length(aug_term[3]) < sub_idx2
    #             sub_idx2 -= 1
    #         end
    #     else # To-Do: probably not required, check and remove.
    #         vals[main_idx] = 0.0
    #         main_idx += 1
    #     end
    # end
end
