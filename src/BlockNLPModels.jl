using JuMP, Random, NLPModels, NLPModelsJuMP, Ipopt, SolverCore, NLPModelsIpopt

include("blk_NLP.jl")
include("dd.jl")
include("Block_Solver.jl")
include("Full_Space.jl")

# Generate a test problem
rng= MersenneTwister(1234)
B = 3 # number of blocks

blk = []
T = rand(rng, 1:5, B)
for i in 1:B
    model = Model()
    @variable(model, x >= 0)
    @objective(model, Min, (x - T[i])^2)
    nlp = MathOptNLPModel(model)
    push!(blk, nlp)
end
A = ones(Float64, B)
b = rand(rng, 100*B:500*B)/100
bm = BlockNLPModel(blk, vcat(A, b), λ0 = 0.0)

# Solve using dual decomposition
dual_decomposition(bm)
# Solve the full-space problem with Ipopt
stats = ipopt(FullSpace(bm))