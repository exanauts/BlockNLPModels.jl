using Random, JuMP
using NLPModels
using NLPModelsJuMP
using NLPModelsIpopt
using BlockNLPModels
using Test
using SparseArrays

include("block_solvers/dual_decomposition.jl")

include("simple_block_model.jl")
include("mpc_problem.jl")
