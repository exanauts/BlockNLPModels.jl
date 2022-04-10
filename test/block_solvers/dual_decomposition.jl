using BlockNLPModels
using NLPModelsIpopt

"""
        dual_decomposition(
            m::AbstractBlockNLPModel; 
            max_iter::Int = 100, 
            step_size::Float64 = 0.1
        )
Implements the dual decomposition algorithm on `BlockNLPModel`.

# Arguments

- `m::AbstractBlockNLPModel`: identifier for the `BlockNLPModel` 
- `max_iter::Int = 100`: total number of dual decomposition iterations (optional) 
- `step_size::Float64 = 0.1`: dual update step size (optional)
"""

function dual_decomposition(m::AbstractBlockNLPModel; max_iter::Int = 100, step_size::Float64 = 0.1)
    N = m.problem_size.block_counter # Number of blocks
    iter_count = 0

    # Initialize the dual variables
    y = zeros(Float64, n_tot_con(m))

    # Initialize an array to store block solutions
    sol = zeros(Float64, m.problem_size.var_counter)

    # Get the linking constraints
    A = get_matrixblocks(m)
    b = get_RHSvector(m)
    while iter_count <= max_iter
        iter_count += 1
        for i in 1:N
            dualized_block = DualizedNLPblockModel(m.blocks[i].problem_block, y, A[i])
            sol[m.blocks[i].var_idx] = ipopt(dualized_block, print_level = 0).solution
        end
        y += step_size.*(sum(A[j]*sol[m.blocks[j].var_idx] for j = 1:N) - b)
    end
    # TODO: Introduce verbosity parameter and make available more output options
    return sol
end  