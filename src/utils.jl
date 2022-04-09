function get_blockmatrix(m::AbstractBlockNLPModel)
    nb = m.problem_size.block_counter
    A = spzeros(m.problem_size.link_counter, m.problem_size.var_counter)
    for i in 1:nb
        for j in 1:length(m.linking_constraints)
            if m.linking_constraints[j]::AbstractLinearLinkConstraint
                A[m.linking_constraints[j].idx, m.blocks[i].var_idx] = 
                m.linking_constraints[j].linking_blocks[i]
            end
        end
    end
    return A
end

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

function get_RHSvector(m::AbstractBlockNLPModel)
    b = zeros(m.problem_size.link_counter)
    for j in 1:length(m.linking_constraints)
        b[m.linking_constraints[j].idx] = b.linking_constraints[j].RHS_vector
    end
    return b
end

function n_tot_con(m::AbstractBlockNLPModel)
    return m.problem_size.link_counter + m.problem_size.con_counter 
end

