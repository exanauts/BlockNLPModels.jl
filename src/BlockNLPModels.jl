module BlockNLPModels

using NLPModels
using NLPModelsJuMP
using Ipopt
using SolverCore
using NLPModelsIpopt


include("blk_NLP.jl")
include("dd.jl")
include("Block_Solver.jl")
include("Full_Space.jl")

export BlockNLPModel, dual_decomposition, FullSpace 

end