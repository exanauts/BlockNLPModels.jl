using BlockNLPModels
using NLPModelsIpopt

"""
        admm(
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
function admm(m::AbstractBlockNLPModel; max_iter::Int = 100, ρ::Float64 = 0.1)
    N = m.problem_size.block_counter # Number of blocks
    iter_count = 0

    # Initialize the dual variables
    y = zeros(Float64, m.problem_size.link_counter)

    # Initialize an array to store primal solutions
    sol = zeros(Float64, m.problem_size.var_counter)

    # Get the linking constraints
    A = get_blockmatrix(m)
    b = get_RHSvector(m)
    while iter_count <= max_iter
        iter_count += 1
        for i in 1:N
            augmented_block = AugmentedNLPBlockModel(m.blocks[i], y, ρ, A, b, sol)
            sol[m.blocks[i].var_idx] = ipopt(augmented_block, print_level = 0).solution
        end
        y += ρ.*(A*sol - b)
    end
    # TODO: Introduce verbosity parameter and make available more output options
    return sol
end  