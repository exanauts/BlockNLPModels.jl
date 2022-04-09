"""
  get_blockmatrix(
    m::AbstractBlockNLPModel 
  )

Returns the concatenated block matrix ``A = [A_1, \\ldots, A_B]``

# Arguments

- `m::AbstractBlockNLPModel`: BlockNLPModel whose block matrix is to be extracted.
"""
function get_blockmatrix(m::AbstractBlockNLPModel)
    nb = m.problem_size.block_counter
    A = spzeros(m.problem_size.link_counter, m.problem_size.var_counter)
    for i in 1:nb
        for j in 1:length(m.linking_constraints)
            # if m.linking_constraints[j] isa AbstractLinearLinkConstraint
                A[m.linking_constraints[j].idx, m.blocks[i].var_idx] = 
                m.linking_constraints[j].linking_blocks[i]
            # end
        end
    end
    return A
end

"""
  get_matrixblocks(
    m::AbstractBlockNLPModel 
  )

Returns a vector of block matrices `[A_1, \\ldots, A_B]``

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel whose block matrices are to be extracted.
"""
function get_matrixblocks(m::AbstractBlockNLPModel)
    nb = m.problem_size.block_counter
    A = Vector{SparseMatrixCSC{Float64, Int}}(undef, nb)
    for i in 1:nb
        A[i] = spzeros(m.problem_size.link_counter, m.blocks[i].meta.nvar)
        for j in 1:length(m.linking_constraints)
            A[i][m.linking_constraints[j].idx, :] = 
            m.linking_constraints[j].linking_blocks[i]
        end
    end
    return A
end

"""
    get_RHSvector(
        m::AbstractBlockNLPModel 
    )

Returns the concatenated RHS vector for all the linking constraints ``b``.

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel whose RHS vector is to be extracted.
"""
function get_RHSvector(m::AbstractBlockNLPModel)
    b = zeros(m.problem_size.link_counter)
    for j in 1:length(m.linking_constraints)
        b[m.linking_constraints[j].idx] = m.linking_constraints[j].RHS_vector
    end
    return b
end

"""
    n_tot_con(
        m::AbstractBlockNLPModel 
    )

Returns the total number of constraints in a BlockNLPModel.

# Arguments

- `m::AbstractBlockNLPModel`: name of the BlockNLPModel.
"""
function n_tot_con(m::AbstractBlockNLPModel)
    return m.problem_size.link_counter + m.problem_size.con_counter 
end

