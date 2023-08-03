#module juliaml

using LinearAlgebra
using LoopVectorization
using Plots
using BenchmarkTools
using Random
using Statistics: mean, std
using JET
using StaticArrays

include("utils.jl")

input_size = 2
output_size = 1
hidden_size = 32
activation = swish
activation_prime = swish_prime

model = MLP(input_size, hidden_size, output_size, activation, activation_prime)

x = randn(Float32, 2, 1000)
y2 = model(x)

display(@benchmark y2 = model(x))

@report_opt model(x)
@report_opt model.layers[1](x)

activation = relu
weights = model.layers[1].weights
bias = model.layers[1].bias

pullback, grads = backward(model.layers[2], rand(Float32, 32, 1000) |> collect, ones(Float32, 32, 1000) |> collect)
display(pullback)




# lmao flux why are u so bad
# i deleted the flux benchmarks but if u want try urselves
# josh doesnt know multivariate calc LMAOO
#end # module juliaml
