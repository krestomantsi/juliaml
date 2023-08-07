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

function gelu(x::Matrix{Float32})::Matrix{Float32}
    pif32 = Float32(pi)
    @. 0.5f0 * x * (1.0f0 + tanh(sqrt(2.0f0 / pif32) * (x + 0.044715f0 * x^3)))
end

function gelu(x)
    pif32 = Float32(pi)
    0.5f0 * x * (1.0f0 + tanh(sqrt(2.0f0 / pif32) * (x + 0.044715f0 * x^3)))
end

# pub fn gelu_prime_scalar(x: f32) -> f32 {
#     let PI = std::f32::consts::PI;
#     let lam = (2.0f32 / PI).sqrt();
#     let a = 0.044715f32;
#     let tanh_term = ((x + a * x.powi(3)) * lam).tanh();
#     //(1 + x (1 + 3 x^2 α) λ Sech[(x + x^3 α) λ]^2 + Tanh[(x + x^3 α) λ])/2}
#     0.5f32
#         * (1.0f32
#             + x * (1.0f32 + 3.0f32 * x.powi(2) * a)
#                 * lam
#                 * sech_scalar(((x + x.powi(3) * a) * lam)).powi(2)
#             + tanh_term)
# }
@inline function gelu_prime(x::Matrix{Float32})::Matrix{Float32}
    pif32 = Float32(pi)
    lam = sqrt(2.0f0 / pif32)
    a = 0.044715f0
    tanh_term = tanh.(lam .* (x .+ a .* x .^ 3))
    0.5f0 .* (1.0f0 .+ x .* (1.0f0 .+ 3.0f0 .* x .^ 2 .* a) .* lam .* sech.(lam .* (x .+ x .^ 3 .* a)) .^ 2 .+ tanh_term)
end

function gelu_prime(x)
    pif32 = Float32(pi)
    lam = sqrt(2.0f0 / pif32)
    a = 0.044715f0
    tanh_term = tanh(lam * (x + a * x^3))
    0.5f0 * (1.0f0 + x * (1.0f0 + 3.0f0 * x^2 * a) * lam * sech(lam * (x + x^3 * a))^2 + tanh_term)
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
    @fastmath @. map(swish, x)
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
    f::Union{typeof(relu),typeof(gelu),typeof(leaky_relu),typeof(none_activation)},
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
    activation::Union{typeof(relu),typeof(gelu),typeof(leaky_relu),typeof(none_activation)}
    activation_prime::Union{typeof(relu_prime),typeof(gelu_prime),typeof(leaky_relu_prime),typeof(none_activation_prime)}
end
# forward call of Dense
function (d::Dense)(x::Matrix{Float32})::Matrix{Float32}
    dense(d.activation, d.weights, d.bias, x)
end


# struct MLP
#     layers::Vector{Dense}
# end

struct MLP
    # layers::Tuple{Vararg{Dense}}
    # layers::Tuple{Dense,Dense,Dense}
    layers::Vector{Dense}
end

@inline function (mlp::MLP)(x::Matrix{Float32})::Matrix{Float32}
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
    # layers::Tuple{Vararg{DenseGradient}}
    layers::Vector{DenseGradient}
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
    # layers = (
    #     Dense(weights1, bias1, activation, activation_prime),
    #     Dense(weights2, bias2, activation, activation_prime),
    #     Dense(weights3, bias3, none_activation, none_activation_prime))
    layers = [
        Dense(weights1, bias1, activation, activation_prime),
        Dense(weights2, bias2, activation, activation_prime),
        Dense(weights3, bias3, none_activation, none_activation_prime)]
    MLP(layers)
end

function backward(d::Dense, x::Matrix{Float32}, pullback::Matrix{Float32})
    m = size(x, 2)
    bias = sum(pullback, dims=2)
    weights = mygem(pullback, x' |> collect)
    #pullback = (d.weights' * pullback) .* d.activation_prime(x)
    pullback = mygem(d.weights' |> collect, pullback) .* d.activation_prime(x)
    grads = DenseGradient(weights, bias)
    return pullback, grads
end

function backward(mlp::MLP, x::Matrix{Float32}, y::Matrix{Float32}, loss_prime::typeof(mse_prime))
    # forward pass
    output::Vector{Matrix{Float32}} = []
    push!(output, x)
    for layer in mlp.layers
        x = layer(x)
        push!(output, x)
    end
    # backward pass
    pullback = loss_prime(output[end], y)
    grads = []
    for i in length(mlp.layers):-1:1
        pullback, grad = backward(mlp.layers[i], output[i], pullback)
        push!(grads, grad)
    end
    return output, MLPGradient(reverse(grads))
end

struct SGDw
    lr::Float32
    weight_decay::Float32
end

function sgd(mlp::MLP, grads::MLPGradient, lr::Float32)
    layers = []
    for ii in 1:length(mlp.layers)
        weights = mlp.layers[ii].weights .- lr * grads.layers[ii].weights
        bias = mlp.layers[ii].bias .- lr * grads.layers[ii].bias
        Dense(weights, bias, mlp.layers[ii].activation, mlp.layers[ii].activation_prime)
        push!(layers, Dense(weights, bias, mlp.layers[ii].activation, mlp.layers[ii].activation_prime))
    end
    MLP(layers)
end

function trainsgd(mlp::MLP, x::Matrix{Float32}, y::Matrix{Float32}, loss::typeof(mse), loss_prime::typeof(mse_prime), epochs, lr)
    for ii in 1:epochs
        outputs, grads = backward(mlp, x, y, loss_prime)
        mlp = sgd(mlp, grads, lr)
        if ii % 1500 == 0
            println("epoch ", ii, " || loss: ", loss(mlp(x), y))
        end
    end
    return mlp
end

function displaynetwork(mlp, x, y, loss_prime)
    outputs, grads = backward(mlp, x, y, loss_prime)
    ind = 1
    for m in mlp.layers
        histogram(m.weights |> vec)
        title!("weights layer " * string(ind) * ".png")
        savefig("weights layer " * string(ind) * ".png")
        histogram(m.bias |> vec)
        title!("bias layer " * string(ind) * ".png")
        savefig("bias layer " * string(ind) * ".png")
        ind += 1
    end
    ind = 0
    for ii in outputs
        histogram(ii |> vec)
        title!("outputs layer " * string(ind) * ".png")
        savefig("outputs layer " * string(ind) * ".png")
        ind += 1
    end
end

function MLP_det(input_size::Int, hidden_size::Int, output_size::Int, activation::Function, activation_prime::Function)
    weights1 = ones(Float32, hidden_size, input_size)
    bias1 = zeros(Float32, hidden_size, 1)
    weights2 = ones(Float32, hidden_size, hidden_size)
    bias2 = zeros(Float32, hidden_size, 1)
    weights3 = ones(Float32, output_size, hidden_size)
    bias3 = zeros(Float32, output_size, 1)
    layers = [
        Dense(weights1, bias1, activation, activation_prime),
        Dense(weights2, bias2, activation, activation_prime),
        Dense(weights3, bias3, none_activation, none_activation_prime)]
    MLP(layers)
end
