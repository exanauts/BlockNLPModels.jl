# Generate a random test problem
rng= MersenneTwister(1234)
B = 3 # number of blocks

block_model = BlockNLPModel()
T = rand(rng, 1:5, B)
for i in 1:B
    model = Model()
    @variable(model, x >= 0)
    @objective(model, Min, (x - T[i])^2)
    @constraint(model, x <= 1)
    nlp = MathOptNLPModel(model)
    add_block(block_model, nlp)
end
A = ones(Float64, B)
b = rand(rng, 50*B:100*B)/100

links = Dict(1=>[A[1]])
for i in 2:B
    links[i] = sparse([A[i]])
end
add_links(block_model, 1, links, b)

@testset "Testing counters" begin
    @test block_model.problem_size.block_counter ≈ B
    @test block_model.problem_size.link_counter ≈ 1
    @test block_model.problem_size.var_counter ≈ B
    @test block_model.problem_size.con_counter ≈ B
end

@testset "Testing BlockNLPModel.jl " begin
    @test length(block_model.blocks) ≈ B
    @test length(block_model.linking_constraints) ≈ 1
    for i in 1:B
        block = block_model.blocks[i]
        @test block.meta.nvar ≈ 1
        @test block.meta.ncon ≈ 1
        @test block.var_idx == i:i
        @test block.con_idx == i:i
        @test block.linking_constraint_id == [1]
    end
    @test block_model.linking_constraints[1].link_map == Dict(1 => [2, 3, 1])
    @test get_linking_matrix(block_model) == sparse([1, 1, 1], [1, 2, 3], [1.0, 1.0, 1.0])
    for i in 1:B
        @test get_linking_matrix_blocks(block_model)[i] == sparse([1], [1], [1.0])
    end
    @test get_rhs_vector(block_model) == [b]
    @test n_constraints(block_model) == 4
end

@testset "Testing FullSpaceModel.jl" begin
    f(x) = sum((x[i] - T[i])^2 for i in 1:B)
    ∇f(x) = [2*(x[i] - T[i]) for i in 1:B]
    g(x) = vcat(x, get_linking_matrix(block_model)*x)
    J(x) = [1.0 0 0;0 1.0 0; 0 0 1.0;1 1 1]
    H(x, y) = [2.0 0 0;0 2.0 0;0 0 2.0]

    fm = FullSpaceModel(block_model)
    @test fm.meta.nvar ≈ B
    @test fm.meta.ncon ≈ B+1
    @test fm.meta.nnzh ≈ B
    @test fm.meta.nnzj ≈ 6

    x = rand(rng, 0:100, B)/100
    y = rand(rng, 0:100, B+1)/100
    @test obj(fm, x) ≈ f(x)
    @test grad(fm, x) ≈ ∇f(x)
    @test cons(fm, x) ≈ g(x)
    @test jac(fm, x) ≈ J(x)
    @test hess(fm, x, y) ≈ H(x, y)

    # stats = ipopt(fm, print_level = 0) # throws an error
    # @test isapprox(stats.solution, solution) 

    # Solve the full-space problem with MadNLP
    # To Do: Why MadNLP throws an error?
    # madnlp(fm)
end

@testset "Testing dualized_block.jl" begin
    A = get_linking_matrix_blocks(block_model)
    y = ones(Float64, 1)
    f(x, i) = (x[1] - T[i])^2 + dot(y, A[i], x)
    ∇f(x, i) = 2*(x[1] - T[i]) .+ A[i]'*y

    for i in 1:B
        x = rand(rng, 0:100, 1)/100
        dualized_block = DualizedNLPBlockModel(block_model.blocks[i].problem_block, y, A[i])
        @test dualized_block.meta.nvar ≈ 1
        @test dualized_block.meta.ncon ≈ 1
        @test obj(dualized_block, x) ≈ f(x, i)
        @test grad(dualized_block, x) ≈ ∇f(x, i)

        # Solve the block using MadNLP
        result = madnlp(dualized_block;print_level=MadNLP.ERROR)
        @test result.solution[1] ≈ 1.0
    end
end

@testset "Testing augmented_block.jl" begin
    A = get_linking_matrix(block_model)
    b = get_rhs_vector(block_model)
    sol = rand(rng, 0:100, B)/100
    ρ = 0.1
    y = ones(Float64, 1)

    f(x, i) = (x[i] - T[i])^2 + dot(y, A[:, block_model.blocks[i].var_idx], x[i]) + (ρ/2)*norm(A*x - b)^2
    ∇f(x, i) = 2*(x[i] - T[i]) .+ (A[:, block_model.blocks[i].var_idx]'*y .+ ρ.*A[:, block_model.blocks[i].var_idx]'*(A*x - b))
    H(x, i) = 2.0 .+ ρ*A[:, block_model.blocks[i].var_idx]'*A[:, block_model.blocks[i].var_idx]

    for i in 1:B
        x = rand(rng, 0:100, 1)/100
        current_sol = deepcopy(sol)
        current_sol[i] = x[1]
        augmented_block = AugmentedNLPBlockModel(block_model.blocks[i], y, ρ, A, b, sol)
        @test augmented_block.meta.nvar ≈ 1
        @test augmented_block.meta.ncon ≈ 1
        @test obj(augmented_block, x) ≈ f(current_sol, i)
        @test grad(augmented_block, x) ≈ ∇f(current_sol, i)
        @test hess(augmented_block, x, [1.0]) ≈ H(x, i) 
    end
end

