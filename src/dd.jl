function dual_decomposition(m::BlockNLP)
    N = length(m.blocks) # Number of blocks
    iter_count = 0
    max_iter = 100

    # Get the intial dual variable value
    y = m.λ

    # Provide dual update step size
    t = 0.1

    # Initialize an array to store block solutions
    x = zeros(Float64, B)

    while iter_count <= max_iter
        iter_count += 1
        for i in 1:N
            x[i] = solve_dualizedblock(m.blocks[i], m.linkconstraints[i], y, print_level = 0).solution[1]
        end
        y += t*(sum(m.linkconstraints[j]*x[j] for j = 1:B) - m.linkconstraints[B+1])
    end
    println("Primal Solution: ", x)
end
