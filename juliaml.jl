#module juliaml

using LinearAlgebra
using LoopVectorization
using Plots
using BenchmarkTools
using Random
using Statistics: mean, std
using JET
using StaticArrays


# almost BLAS level speed by just doing a silly @turbo
function mygemmavx!(C::Matrix{Float32}, A::Matrix{Float32}, B::Matrix{Float32})
    @turbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end

function mygem(A::Matrix{Float32}, B::Matrix{Float32})::Matrix{Float32}
    # will undefined make a diff?
    C = Matrix{Float32}(undef, size(A, 1), size(B, 2))
    mygemmavx!(C, A, B)
    return C
end


# each layer will have a struct definition
# a custom pullback function
# and ofc a forward call
# ideally i wanted static arrays here but the compilation takes for ever
struct Dense
    weights::Matrix{Float32}
    bias::Matrix{Float32}
    activation::Function
    activation_prime::Function
end


# forward call of Dense
function (d::Dense)(x::Matrix{Float32})::Matrix{Float32}
    output = mygem(d.weights, x) .+ d.bias
    d.activation(output)
end

# default gemm
# function (d::Dense)(x::Matrix{Float32})
#     output = d.weights * x .+ d.bias
#     d.activation(output)
# end

# struct MLP
#     layers::Vector{Dense}
# end

struct MLP
    layers::Tuple{Vararg{Dense}}
end

function (mlp::MLP)(x::Matrix{Float32})::Matrix{Float32}
    output = x
    @inbounds @fastmath for layer in mlp.layers
        output = layer(output)
        # dont even ask me why i do this
        # println(size(output))
    end
    output
end

#
struct DenseGradient
    weights::Matrix{Float32}
    bias::Matrix{Float32}
end

struct MLPGradient
    layers::Tuple{Vararg{DenseGradient}}
end

function relu(x::Matrix{Float32})::Matrix{Float32}
    max.(x, 0.0f0)
end

function relu_prime(x::Matrix{Float32})::Matrix{Float32}
    x .> 0.0f0
end

function swish_scalar(x::Float32)::Float32
    @fastmath x / (1.0f0 + exp(-x))
end

function swish(x::Matrix{Float32})::Matrix{Float32}
    swish_scalar.(x)
end

function swish_prime(x::Matrix{Float32})::Matrix{Float32}
    @fastmath swish(x) .+ (1.0f0 .- swish(x)) .* exp.(-x) ./ (1.0f0 .+ exp.(-x))
end

function none_activation(x::Matrix{Float32})::Matrix{Float32}
    x
end

function none_activation_prime(x::Matrix{Float32})::Matrix{Float32}
    ones(eltype(x), size(x))
end

function mse(x::Matrix{Float32}, y::Matrix{Float32})::Float32
    sum((x .- y) .^ 2) / (size(x, 2) |> Float32)
end

function mse_prime(x::Matrix{Float32}, y::Matrix{Float32})::Matrix{Float32}
    2.0f0 .* (x .- y) ./ (size(x, 2) |> Float32)
end


function MLP(input_size::Int, hidden_size::Int, output_size::Int, activation::Function, activation_prime::Function)
    weights1 = randn(Float32, hidden_size, input_size) ./ sqrt(hidden_size)
    bias1 = zeros(Float32, hidden_size, 1)
    weights2 = randn(Float32, hidden_size, hidden_size) ./ sqrt(hidden_size)
    bias2 = zeros(Float32, hidden_size, 1)
    weights3 = randn(Float32, output_size, hidden_size) ./ sqrt(hidden_size)
    bias3 = zeros(Float32, output_size, 1)
    layers = (
        Dense(weights1, bias1, activation, activation_prime),
        Dense(weights2, bias2, activation, activation_prime),
        Dense(weights3, bias3, none_activation, none_activation_prime))
    MLP(layers)
end

model = MLP(2, 32, 1, swish, swish_prime)

x = randn(Float32, 2, 1000)
y2 = model(x)

display(@benchmark y2 = model(x))

# still runtime dispatch selected on dense forward
# cant figure it out please find it
@report_opt model(x)
@report_opt model.layers[1](x)

activation = swish
weights = model.layers[1].weights
bias = model.layers[1].bias

# dense(weights, x, bias, activation)
# lmao flux why are u so bad
# i deleted the flux benchmarks but if u want try urselves
# josh doesnt know multivariate calc LMAOO
#end # module juliaml
