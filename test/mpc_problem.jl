using LinearAlgebra

# Linearized discrete state-space model with a sampling time of 1 min
A = [0.2681 -0.00338 -0.00728;
    9.703 0.3279 -25.44;
    0 0 1];
B = [-0.00537 0.1655;
    1.297 97.91;
    0 -6.637];
Bp = [-0.1175;69.74;6.637]

# Tuning matrices
Q = [1/(0.4)^2 0 0;
    0 200/(100)^2 0;
    0 0 1/(0.2)^2]
R = [1/(20)^2 0;
    0 1/(0.01)^2]

# Set the length of prediction horizon
N = 50
# Set the disturbance value
w = 10

# Set the intial state
x0 = [0;0;0]

m = 2 # number of inputs
n = 3 # number of states
u_l = -5
u_u = 5
x_l = -5
x_u = 5
G = spzeros(N*n, N*(n+m))
F = spzeros(N*n, n)

G[1:n, 1:m+n] = hcat(-B, I)
for i = 1:N-1
    G[i*n+1:i*n+n, (m+n)*i-n+1:(m+n)*i+n+m] = hcat(-A, -B, I)
end
F[1:n, 1:n] = A

# Initialize an empty block NLP model
block_mpc = BlockNLPModel()

# Fill the model with NLP blocks
for i in 1:2N
    if i%2 != 0
        mpc_block = Model()
        @variables(mpc_block, begin
            u_l <= u[1:m] <= u_u
        end)
        @objective(mpc_block, Min, dot(u, R, u))
        nlp = MathOptNLPModel(mpc_block)
        add_block(block_mpc, nlp)
    else
        mpc_block = Model()
        @variables(mpc_block, begin
            x_l <= x[1:n] <= x_u
        end)
        @objective(mpc_block, Min, dot(x, Q, x))
        nlp = MathOptNLPModel(mpc_block)
        add_block(block_mpc, nlp)
    end
end

# Add linking constraints all at once
links = Dict(1=>Matrix(G[:, 1:m]))
links[2] = Matrix(G[:, m+1:m+n])
count = 3
for i = 2:N
    for j = 1:2
        global count
        if count%2 != 0
            links[count] = Matrix(G[:, (i-1)*(m+n)+1:(i-1)*(m+n)+m])
            count += 1
        else
            links[count] = Matrix(G[:, (i-1)*(m+n)+m+1:(i-1)*(m+n)+m+n])
            count += 1
        end
    end
end
add_links(block_mpc, N*n, links, F*x0)
fs = FullSpaceModel(block_mpc)

# Solve using dual decomposition
solution = dual_decomposition(block_mpc)

