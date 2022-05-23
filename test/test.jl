rng = MersenneTwister(1234)
B = 3 # number of blocks

block_model = BlockNLPModel()
T = rand(rng, 1:5, B)
for i = 1:B
    model = Model()
    @variable(model, x >= 0, start = rand(rng))
    @objective(model, Min, (x - T[i])^2)
    @constraint(model, x <= 1)
    nlp = MathOptNLPModel(model)
    add_block(block_model, nlp)
end
A = ones(Float64, B)
b = rand(rng, 50*B:100*B) / 100

links = Dict(1 => [A[1]])
for i = 2:B
    links[i] = sparse([A[i]])
end
add_links(block_model, 1, links, b)
fm = FullSpaceModel(block_model)
