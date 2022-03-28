using Random, JuMP
using NLPModels
using NLPModelsJuMP
using NLPModelsIpopt
using BlockNLPModels
using Test

include("block_solvers/dual_decomposition.jl")

# Generate a test problem
rng= MersenneTwister(1234)
B = 3 # number of blocks

blocks = Array{AbstractNLPModel,1}(undef, 0)

T = rand(rng, 1:5, B)
for i in 1:B
    model = Model()
    @variable(model, x >= 0)
    @objective(model, Min, (x - T[i])^2)
    nlp = MathOptNLPModel(model)
    push!(blocks, nlp)
end
A = ones(Float64, B)
b = rand(rng, 100*B:500*B)/100
block_model = BlockNLPModel(blocks, vcat(A, b))

# Solve using dual decomposition
solution = dual_decomposition(block_model)

# Solve the full-space problem with Ipopt
stats = ipopt(FullSpaceModel(block_model), print_level = 0)

@test isapprox(stats.solution, solution) 