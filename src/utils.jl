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
    C = zeros(eltype(A), size(A, 1), size(B, 2))
    mygemmavx!(C, A, B)
    return C
end

function relu(x)
    max(x, zero(eltype(x)))
end

function relu(x::Matrix{Float32})::Matrix{Float32}
    max.(x, 0.0f0)
end

function relu_prime(x::Matrix{Float32})::Matrix{Float32}
    x .> 0.0f0
end

leaky_relu(x) = max(x, 0.01f0 * x)

function leaky_relu(x::Matrix{Float32})::Matrix{Float32}
    max.(x, 0.01f0 .* x)
end

function leaky_relu_prime(x::Matrix{Float32})::Matrix{Float32}
    ifelse.(x .> 0.0f0, 1.0f0, 0.01f0)
end


function none_activation(x::Matrix{Float32})::Matrix{Float32}
    x
end
function none_activation(x)
    x
end

function none_activation_prime(x::Matrix{Float32})::Matrix{Float32}
    ones(eltype(x), size(x))
end

function swish(x::Float32)::Float32
    x / (1.0f0 + exp(-x))
end
function swish(x::Matrix{Float32})::Matrix{Float32}
    @. map(swish, x)
end

function swish_prime(x::Matrix{Float32})::Matrix{Float32}
    @fastmath swish(x) .+ (1.0f0 .- swish(x)) .* exp.(-x) ./ (1.0f0 .+ exp.(-x))
end

# each layer will have a struct definition
# a custom pullback function
# and ofc a forward call
# ideally i wanted static arrays here but the compilation takes for ever

# taken from simple chains
@inline function dense(
    f::Union{typeof(relu),typeof(swish),typeof(leaky_relu),typeof(none_activation)},
    W::Matrix{Float32},
    b::Matrix{Float32},
    x::Matrix{Float32}
)
    C = zeros(eltype(x), size(W, 1), size(x, 2))
    @turbo for m ∈ axes(W, 1), n ∈ axes(x, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(W, 2)
            Cmn += W[m, k] * x[k, n]
        end
        C[m, n] = f(Cmn + b[m])
    end
    return C
end

struct Dense
    weights::Matrix{Float32}
    bias::Matrix{Float32}
    activation::Union{typeof(relu),typeof(swish),typeof(leaky_relu),typeof(none_activation)}
    activation_prime::Union{typeof(relu_prime),typeof(swish_prime),typeof(leaky_relu_prime),typeof(none_activation_prime)}
end
# forward call of Dense
function (d::Dense)(x::Matrix{Float32})::Matrix{Float32}
    dense(d.activation, d.weights, d.bias, x)
end


# struct MLP
#     layers::Vector{Dense}
# end

struct MLP
    layers::Tuple{Vararg{Dense}}
    # layers::Tuple{Dense,Dense,Dense}
end

function (mlp::MLP)(x::Matrix{Float32})::Matrix{Float32}
    output = x
    for layer in mlp.layers
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


function mse(x::Matrix{Float32}, y::Matrix{Float32})::Float32
    sum((x .- y) .^ 2) / (size(x, 2) |> Float32)
end

function mse_prime(x::Matrix{Float32}, y::Matrix{Float32})::Matrix{Float32}
    2.0f0 .* (x .- y) ./ (size(x, 2) |> Float32)
end


function MLP(input_size::Int, hidden_size::Int, output_size::Int, activation::Function, activation_prime::Function)
    weights1 = (randn(Float32, hidden_size, input_size) ./ (sqrt(hidden_size) |> Float32))
    bias1 = zeros(Float32, hidden_size, 1)
    weights2 = (randn(Float32, hidden_size, hidden_size) ./ (sqrt(hidden_size) |> Float32))
    bias2 = zeros(Float32, hidden_size, 1)
    weights3 = (randn(Float32, output_size, hidden_size) ./ (sqrt(hidden_size) |> Float32))
    bias3 = zeros(Float32, output_size, 1)
    layers = (
        Dense(weights1, bias1, activation, activation_prime),
        Dense(weights2, bias2, activation, activation_prime),
        Dense(weights3, bias3, none_activation, none_activation_prime))
    MLP(layers)
end

function backward(d::Dense, x::Matrix{Float32}, pullback::Matrix{Float32})
    bias = pullback |> copy
    weights = mygem(pullback, x' |> collect)
    pullback = mygem(d.weights' |> collect, pullback) .* d.activation_prime(x)
    grads = DenseGradient(weights, bias)
    return pullback, grads
end
