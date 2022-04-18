"""
  get_linking_matrix(
    m::AbstractBlockNLPModel 
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
    A = [spzeros(m.problem_size.link_counter, m.blocks[i].meta.nvar) for i in 1:nb]
    for i in 1:nb
        for j in 1:length(m.linking_constraints)
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
    for j in 1:length(m.linking_constraints)
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

