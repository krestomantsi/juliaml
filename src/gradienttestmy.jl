using LinearAlgebra
using LoopVectorization
using Plots
using BenchmarkTools
using Random
using Statistics: mean, std
using JET

include("utils.jl")

input_size = 1
output_size = 1
hidden_size = 16
activation = gelu
activation_prime = gelu_prime
epochs = 80000
lr = 0.01f0

model = MLP_det(input_size, hidden_size, output_size, activation, activation_prime)

#x = randn(Float32, input_size, 1000)
x = LinRange(-1, 1, 20)' |> collect .|> Float32

y = sin.(2 * Float32(pi) * x)
# y = x .^ 2
y2 = model(x)



pullback, grads = backward(model, x, y, mse_prime)

ii = 2
display(grads.layers[ii].weights)
display(grads.layers[ii].bias)
