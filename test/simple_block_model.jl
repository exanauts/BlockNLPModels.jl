# Generate a test problem
rng= MersenneTwister(1234)
B = 3 # number of blocks

block_model = BlockNLPModel()
T = rand(rng, 1:5, B)
for i in 1:B
    model = Model()
    @variable(model, x >= 0)
    @objective(model, Min, (x - T[i])^2)
    nlp = MathOptNLPModel(model)
    add_block(block_model, nlp)
end
A = ones(Float64, 1, B)
A = sparse(A)
b = rand(rng, 100*B:500*B, 1)/100

links = Dict(1=>sparse(reshape([A[1]], (1,1))))
for i in 2:B
    links[i] = sparse(reshape([A[1]], (1,1)))
end
add_links(block_model, 1, links, sparse(b))
# Solve using dual decomposition
solution = dual_decomposition(block_model)

# # Solve the full-space problem with Ipopt
stats = ipopt(FullSpaceModel(block_model), print_level = 0)

@test isapprox(stats.solution, solution)
