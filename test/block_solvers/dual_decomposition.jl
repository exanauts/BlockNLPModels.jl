using BlockNLPModels
using NLPModelsIpopt

function dual_decomposition(m::AbstractBlockNLPModel)
    N = length(m.blocks) # Number of blocks
    iter_count = 0
    max_iter = 100

    # Get the intial dual variable value
    y = 0.0

    # Provide dual update step size
    t = 0.1

    # Initialize an array to store block solutions
    x = zeros(Float64, B)

    while iter_count <= max_iter
        iter_count += 1
        for i = 1:N
            dualized_block = DualizedNLPblockModel(m.blocks[i], [y], [m.linkconstraints[i]])
            x[i] = ipopt(dualized_block, print_level = 0).solution[1]
        end
        y += t * (sum(m.linkconstraints[j] * x[j] for j = 1:B) - m.linkconstraints[B+1])
    end
    # TODO: Introduce verbosity parameter and make available more output options
    return x
end
