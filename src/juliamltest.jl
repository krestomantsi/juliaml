module juliaml

using LinearAlgebra
using LoopVectorization
using Plots
using BenchmarkTools
using Random
using Statistics: mean, std
using JET
using Base.Threads: Threads, @spawn
using DataFrames
using CSV
using JSON
using PrecompileTools
# testing using
using juliaml


input_size = 1
output_size = 1
hidden_size = 32
activation = swish
activation_prime = swish_prime
epochs = 30_000
lr = 0.01f0
wd = 0.00001f0
n = 100

model = MLP(input_size, hidden_size, output_size, activation, activation_prime)

x = LinRange(-1, 1, n)' |> collect .|> Float32

y = sin.(4 * Float32(pi) * x)
#y = cos.(3 * Float32(pi) * x) .^ 11

y2 = model(x)

@time model = train!(model, x, y, lr, wd, epochs, mse, mse_prime, false)

displaynetwork(model, x, y, mse_prime)

# testing saving & loading
save(model, "model.json")
model2 = loadmlp("model.json")

x2 = LinRange(-1.2, 1.2, 200)' |> collect .|> Float32
y2 = model2(x2)
scatter(x', y', label="data")
display(plot!(x2', y2', label="model"))
savefig("result.png")


# testing parallel forward
# function parallel_backward(mlp::MLP, x::Matrix{Float32}, y::Matrix{Float32}, loss_prime)
# end
end
