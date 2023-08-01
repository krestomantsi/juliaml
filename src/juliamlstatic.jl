#module juliaml

using LinearAlgebra
using LoopVectorization
using Plots
using BenchmarkTools
using Random
using Statistics: mean, std
using JET
using StaticArrays


struct Dense
    weights
    bias
    activation::Function
    activation_prime::Function
end

function (d::Dense)(x)
    output = d.weights * x .+ d.bias
    d.activation(output)
end


struct MLP
    layers::Vector{Dense}
end


function (mlp::MLP)(x)
    output = x
    @inbounds for layer in mlp.layers
        output = layer(output)
        # println(size(output))
    end
    output
end

function (mlp::MLP)(x)
    output = x
    @inbounds for layer in mlp.layers
        output = layer(output)
        # println(size(output))
    end
    output
end

# struct DenseGradient
#     weights::Array{Float32,2}
#     bias::Array{Float32,2}
# end

struct DenseGradient
    weights::SMatrix
    bias::SMatrix
end

struct MLPGradient
    layers::Vector{DenseGradient}
end

function relu(x)
    max.(x, 0.0f0)
end

function relu_prime(x)
    x .> 0.0f0
end

function swish_scalar(x::Float32)::Float32
    @fastmath x / (1.0f0 + exp(-x))
end

function swish(x)
    swish_scalar.(x)
end

function swish_prime(x)
    @fastmath swish(x) .+ (1.0f0 .- swish(x)) .* exp.(-x) ./ (1.0f0 .+ exp.(-x))
end

function none_activation(x)
    x
end

function none_activation_prime(x)
    ones(eltype(x), size(x))
end

function mse(x, y)
    sum((x .- y) .^ 2) / (size(x, 2) |> Float32)
end

function mse_prime(x, y)
    2.0f0 .* (x .- y) ./ (size(x, 2) |> Float32)
end

function MLP(input_size::Int, hidden_size::Int, output_size::Int, activation::Function, activation_prime::Function)
    weights1 = SMatrix{hidden_size,input_size}(randn(Float32, hidden_size, input_size) ./ (sqrt(hidden_size) |> Float32))
    bias1 = SMatrix{hidden_size,1}(zeros(Float32, hidden_size, 1))
    weights2 = SMatrix{hidden_size,hidden_size}(randn(Float32, hidden_size, hidden_size) ./ (sqrt(hidden_size) |> Float32))
    bias2 = SMatrix{hidden_size,1}(zeros(Float32, hidden_size, 1))
    weights3 = SMatrix{output_size,hidden_size}(randn(Float32, output_size, hidden_size) ./ (sqrt(hidden_size) |> Float32))
    bias3 = SMatrix{output_size,1}(zeros(Float32, output_size, 1))
    layers = Vector{Dense}()
    # push!(layers, Dense(hidden_size, input_size, weights1, bias1, activation, activation_prime))
    # push!(layers, Dense(hidden_size, hidden_size, weights2, bias2, activation, activation_prime))
    # push!(layers, Dense(output_size, hidden_size, weights3, bias3, none_activation, none_activation_prime))
    push!(layers, Dense(weights1, bias1, activation, activation_prime))
    push!(layers, Dense(weights2, bias2, activation, activation_prime))
    push!(layers, Dense(weights3, bias3, none_activation, none_activation_prime))
    MLP(layers)
end

input_size = 1
output_size = 1
hidden_size = 32
activation = relu
activation_prime = relu_prime
model = MLP(1, 32, 1, swish, swish_prime)

x = randn(Float32, 1, 1000) |> SMatrix{1,1000}
y = sin.(x)
y2 = model(x)

# display(@benchmark y2 = model(x))

# @report_opt model(x)
# @report_opt model.layers[1](x)


# lmao flux why are u so bad
#end # module juliaml
