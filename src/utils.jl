# almost BLAS level speed by just doing a silly @turbo
@inline function mygemmavx!(C::Matrix{Float32}, A::Matrix{Float32}, B::Matrix{Float32})
    @turbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end

@inline function mygem(A::Matrix{Float32}, B::Matrix{Float32})::Matrix{Float32}
    C = zeros(eltype(A), size(A, 1), size(B, 2))
    mygemmavx!(C, A, B)
    return C
end

@inline function relu(x)
    max(x, zero(eltype(x)))
end

function relu(x::Matrix{Float32})::Matrix{Float32}
    vmap(relu, x)
end
@inline relu_prime(x) = x .> 0.0f0

function relu_prime(x::Matrix{Float32})::Matrix{Float32}
    vmap(relu_prime, x)
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


function gelu_prime(x::Matrix{Float32})::Matrix{Float32}
    pif32 = Float32(pi)
    lam = sqrt(2.0f0 / pif32)
    a = 0.044715f0
    tanh_term = tanh.(lam .* (x .+ a .* x .^ 3))
    0.5f0 .* (1.0f0 .+ x .* (1.0f0 .+ 3.0f0 .* x .^ 2 .* a) .* lam .* sech.(lam .* (x .+ x .^ 3 .* a)) .^ 2 .+ tanh_term)
end

function gelu(x)::Float32
    pif32 = Float32(pi)
    0.5f0 * x * (1.0f0 + tanh(sqrt(2.0f0 / pif32) * (x + 0.044715f0 * x^3)))
end

function gelu_prime(x)::Float32
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

@inline function swish(x)
    x / (1.0f0 + exp(-x))
end

function swish(x::Matrix{Float32})::Matrix{Float32}
    vmap(swish, x)
end

@inline function swish_prime(x)
    dumpa = 1.0f0 + exp(-x)
    (x * dumpa + dumpa - x) / dumpa^2
end

function swish_prime(x::AbstractMatrix)::AbstractMatrix
    vmap(swish_prime, x)
end

# each layer will have a struct definition
# a custom pullback function
# and ofc a forward call
# ideally i wanted static arrays here but the compilation takes for ever

# taken from simple chains
@inline function dense(
    f::Union{typeof(relu),typeof(gelu),typeof(leaky_relu),typeof(swish),typeof(none_activation)},
    W::Matrix{Float32},
    b::Matrix{Float32},
    x::Matrix{Float32}
)::Matrix{Float32}
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

struct TurboDense
    weights::Matrix{Float32}
    bias::Matrix{Float32}
    activation::Union{typeof(relu),typeof(gelu),typeof(swish),typeof(leaky_relu),typeof(none_activation)}
    activation_prime::Union{typeof(relu_prime),typeof(gelu_prime),typeof(swish_prime),typeof(leaky_relu_prime),typeof(none_activation_prime)}
end

# forward call of Dense
function (d::TurboDense)(x::Matrix{Float32})::Matrix{Float32}
    dense(d.activation, d.weights, d.bias, x)
end

struct TurboLayerNorm
    epsilon::Float32
end

struct TurboNorm
    epsilon::Float32
    mean::Matrix{Float32}
    std::Matrix{Float32}
end

# forward call of LayerNorm
@inline function (d::TurboLayerNorm)(x::Matrix{Float32})::Matrix{Float32}
    eps = d.epsilon
    xmean = mean(x, dims=1)
    xstd = std(x, dims=1, mean=xmean, corrected=false)
    return @. (x - xmean) / (xstd + eps)
end

function (d::TurboNorm)(x::Matrix{Float32})::Matrix{Float32}
    eps = d.epsilon
    xmean = d.mean
    xstd = d.std
    return @. (x - xmean) / (xstd + eps)
end

struct MLP
    layers::Vector{Union{TurboNorm,TurboDense,TurboLayerNorm}}
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

mutable struct TurboDenseGradient
    weights::Matrix{Float32}
    bias::Matrix{Float32}
end

struct MLPGradient
    layers::Vector{TurboDenseGradient}
end


function mse(x::Matrix{Float32}, y::Matrix{Float32})::Float32
    sum((x .- y) .^ 2) / (size(x, 2) |> Float32)
end

function mse_prime(x::Matrix{Float32}, y::Matrix{Float32})::Matrix{Float32}
    2.0f0 .* (x .- y) ./ (size(x, 2) |> Float32)
end


function MLP(input_size::Int, hidden_size::Int, output_size::Int, activation::Function, activation_prime::Function)::MLP
    weights1 = (randn(Float32, hidden_size, input_size) ./ (sqrt(hidden_size) |> Float32))
    bias1 = zeros(Float32, hidden_size, 1)
    weights2 = (randn(Float32, hidden_size, hidden_size) ./ (sqrt(hidden_size) |> Float32))
    bias2 = zeros(Float32, hidden_size, 1)
    weights3 = (randn(Float32, output_size, hidden_size) ./ (sqrt(hidden_size) |> Float32))
    bias3 = zeros(Float32, output_size, 1)
    # layers = (
    #     TurboDense(weights1, bias1, activation, activation_prime),
    #     Dense(weights2, bias2, activation, activation_prime),
    #     Dense(weights3, bias3, none_activation, none_activation_prime))
    layers = [
        TurboDense(weights1, bias1, activation, activation_prime),
        TurboDense(weights2, bias2, activation, activation_prime),
        TurboDense(weights3, bias3, none_activation, none_activation_prime)]
    MLP(layers)
end

@inline function backward(d::TurboDense, x::Matrix{Float32}, z::Matrix{Float32}, pullback::Matrix{Float32})::Tuple{Matrix{Float32},TurboDenseGradient}
    #m = size(x, 2) |> Float32
    dz = pullback .* d.activation_prime(z)
    bias = sum(dz, dims=2)
    weights = mygem(dz, x' |> collect)
    pullback = mygem(d.weights' |> collect, dz)
    grads = TurboDenseGradient(weights, bias)
    return pullback, grads
end

function backward(mlp::MLP, x::Matrix{Float32}, y::Matrix{Float32}, loss_prime::typeof(mse_prime))::Tuple{Vector{Matrix{Float32}},MLPGradient}
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
        pullback, grad = backward(mlp.layers[i], output[i], output[i+1], pullback)
        push!(grads, grad)
    end
    return output, MLPGradient(reverse(grads))
end

struct SGDw
    lr::Float32
    weight_decay::Float32
end

# struct Adam
#     lr::AbstractFloat
#     lambda::AbstractFloat
#     beta1::AbstractFloat
#     beta2::AbstractFloat
#     # to be continued
# end

function sgd(mlp::MLP, grads::MLPGradient, lr::Float32)
    layers = []
    for ii in 1:length(mlp.layers)
        weights = mlp.layers[ii].weights .- lr * grads.layers[ii].weights
        bias = mlp.layers[ii].bias .- lr * grads.layers[ii].bias
        TurboDense(weights, bias, mlp.layers[ii].activation, mlp.layers[ii].activation_prime)
        push!(layers, TurboDense(weights, bias, mlp.layers[ii].activation, mlp.layers[ii].activation_prime))
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

function MLP_det(input_size::Int, hidden_size::Int, hidden_size2, output_size::Int, activation::Function, activation_prime::Function)
    weights1 = ones(Float32, hidden_size, input_size)
    bias1 = zeros(Float32, hidden_size, 1)
    weights2 = ones(Float32, hidden_size2, hidden_size)
    bias2 = zeros(Float32, hidden_size2, 1)
    weights3 = ones(Float32, output_size, hidden_size2)
    bias3 = zeros(Float32, output_size, 1)
    layers = [
        TurboDense(weights1, bias1, activation, activation_prime),
        TurboDense(weights2, bias2, activation, activation_prime),
        TurboDense(weights3, bias3, none_activation, none_activation_prime)]
    MLP(layers)
end

# function tmapreduce(f, op, itr; tasks_per_thread::Int=2, kwargs...)
#     chunk_size = max(1, length(itr) ÷ (tasks_per_thread * nthreads()))
#     tasks = map(Iterators.partition(itr, chunk_size)) do chunk
#         @spawn mapreduce(f, op, chunk; kwargs...)
#     end
#     mapreduce(fetch, op, tasks; kwargs...)
# end


mutable struct AdamState
    lr::AbstractFloat
    lambda::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    epsilon::AbstractFloat
    m::MLPGradient
    v::MLPGradient
    t::Int
end

@inline function fmap(mlp::MLPGradient, f)::MLPGradient
    layers = []
    @inbounds for ii in 1:length(mlp.layers)
        weights = f.(mlp.layers[ii].weights)
        bias = f.(mlp.layers[ii].bias)
        push!(layers, TurboDenseGradient(weights, bias))
    end
    MLPGradient(layers)
end

function adam_init(grads::MLPGradient, lr, lambda, beta1, beta2)::AdamState
    m = fmap(grads, x -> 0.0f0)
    # println("m ", m)
    v = fmap(grads, x -> 0.0f0)
    AdamState(lr, lambda, beta1, beta2, 1.0f-8, m, v, 0)
end

@inline function adamw!(mlp::MLP, grads::MLPGradient, adam::AdamState)#::MLP
    adam.t = adam.t + 1
    b::Float32 = adam.beta1
    b2::Float32 = adam.beta2
    b11 = 1.0f0 - b
    b22 = 1.0f0 - b2
    layers::Vector{TurboDense} = []
    @inbounds for ii in 1:length(mlp.layers)
        mw = @turbo @. b * adam.m.layers[ii].weights + b11 * grads.layers[ii].weights
        mb = @turbo @. b * adam.m.layers[ii].bias + b11 * grads.layers[ii].bias
        vw = @turbo @. b2 * adam.v.layers[ii].weights + b22 * grads.layers[ii].weights^2
        vb = @turbo @. b2 * adam.v.layers[ii].bias + b22 * grads.layers[ii].bias^2
        mhatw = @turbo @. mw / (1.0f0 - b^adam.t)
        mhatb = @turbo @. mb / (1.0f0 - b^adam.t)
        vhatw = @turbo @. sqrt(vw / (1.0f0 - b2^adam.t))
        vhatb = @turbo @. sqrt(vb / (1.0f0 - b2^adam.t))
        weights = @turbo @. mlp.layers[ii].weights - adam.lr * mhatw / (vhatw + adam.epsilon) - adam.lambda * mlp.layers[ii].weights
        bias = @turbo @. mlp.layers[ii].bias - adam.lr * mhatb / (vhatb + adam.epsilon) - adam.lambda * mlp.layers[ii].bias
        @turbo adam.m.layers[ii].weights = mw # |> copy
        @turbo adam.m.layers[ii].bias = mb # |> copy
        @turbo adam.v.layers[ii].weights = vw # |> copy
        @turbo adam.v.layers[ii].bias = vb # |> copy
        #push!(layers, TurboDense(weights, bias, mlp.layers[ii].activation, mlp.layers[ii].activation_prime))
        @turbo mlp.layers[ii].bias .= bias
        @turbo mlp.layers[ii].weights .= weights
    end
    #MLP(layers)
end

@inline function chunk(x::Matrix{Float32})::Vector{Matrix{Float32}}
    batch_size = 32
    x_tuple = []
    for ii in 1:batch_size:size(x, 2)
        if ii + batch_size - 1 > size(x, 2)
            push!(x_tuple, x[:, ii:end])
        else
            push!(x_tuple, x[:, ii:ii+batch_size-1])
        end
    end
    return x_tuple
end

function train!(mlp::MLP, x::Matrix{Float32}, y::Matrix{Float32}, lr::Float32, wd::Float32, epochs::Int, loss::Function, loss_prime::Function, parallel::Bool)
    _outputs, grads = backward(mlp, x, y, loss_prime)
    opt = adam_init(grads, lr, wd, 0.9f0, 0.999f0)
    if parallel == false
        @inbounds for ii in 1:epochs
            _outputs, grads = backward(mlp, x, y, loss_prime)
            #mlp = adamw!(mlp, grads, opt)
            adamw!(mlp, grads, opt)
            if ii % 1500 == 0
                println("epoch ", ii, " || loss: ", loss(mlp(x), y))
            end
        end
    else
        batch_size = 32
        x_tuple = chunk(x, batch_size)
        y_tuple = chunk(y, batch_size)
        nbatch = length(x_tuple) |> Float32
        @inbounds for ii in 1:epochs
            _outputs, grads = Folds.mapreduce((x, y) -> backward(mlp, x, y, loss_prime)[2], +, x_tuple, y_tuple, ThreadedEx())
            grads = fmap(grads, x -> x / nbatch)
            adamw!(mlp, grads, opt)
            if ii % 1500 == 0
                println("epoch ", ii, " || loss: ", loss(mlp(x), y))
            end
        end
    end
    return mlp
end


function writedict!(m::TurboDense, dic, seq::Integer)
    wsz = m.weights |> size
    bsz = m.bias |> size
    weights = m.weights |> vec
    bias = m.bias |> vec
    dic["layer_"*string(seq)*"_type"] = "dense"
    dic["layer_"*string(seq)*"_weights"] = weights
    dic["layer_"*string(seq)*"_bias"] = bias
    dic["layer_"*string(seq)*"_activation"] = m.activation |> string
    dic["layer_"*string(seq)*"_activation_prime"] = m.activation_prime |> string
    dic["layer_"*string(seq)*"_weight_size"] = wsz
    dic["layer_"*string(seq)*"_bias_size"] = bsz
end

function writedict!(m::TurboNorm, dic, seq::Integer)
    xmean = m.mean |> vec
    xstd = m.std |> vec
    sh = size(xmean, 1)
    eps = m.epsilon
    dic["layer_"*string(seq)*"_type"] = "norm"
    dic["layer_"*string(seq)*"_eps"] = eps
    dic["layer_"*string(seq)*"_mean"] = xmean
    dic["layer_"*string(seq)*"_std"] = xstd
end

function writedict!(m::TurboLayerNorm, dic, seq::Integer)
    eps = m.epsilon
    dic["layer_"*string(seq)*"_type"] = "layernorm"
    dic["layer_"*string(seq)*"_eps"] = eps
end

"""
    save(model::MLP, fname::string)

    save the model to a json file
"""
function save(model::MLP, fname::String)
    n = model.layers |> length
    dic = Dict()
    dic["n_layers"] = n
    for ii in 1:n
        writedict!(model.layers[ii], dic, ii)
    end
    jdic = json(dic)
    JSON.write(fname, jdic)
end

"""
    loadmlp(fname::string)

    load a Turbo model from a json file
"""
function loadmlp(fname::String)::MLP
    dic = JSON.parsefile(fname; dicttype=Dict, inttype=Int64, use_mmap=true)
    n = dic["n_layers"]
    layers = []
    for ii in 1:n
        type = dic["layer_"*string(ii)*"_type"]
        if type == "dense"
            weights = dic["layer_"*string(ii)*"_weights"] .|> Float32
            bias = dic["layer_"*string(ii)*"_bias"] .|> Float32
            nw1 = dic["layer_"*string(ii)*"_weight_size"][1] |> Int64
            nw2 = dic["layer_"*string(ii)*"_weight_size"][2] |> Int64
            nb1 = dic["layer_"*string(ii)*"_bias_size"][1] |> Int64
            nb2 = dic["layer_"*string(ii)*"_bias_size"][2] |> Int64
            activation = dic["layer_"*string(ii)*"_activation"]
            activation_prime = dic["layer_"*string(ii)*"_activation_prime"]

            if activation == "relu"
                activation = relu
                activation_prime = relu_prime
            elseif activation == "gelu"
                activation = gelu
                activation_prime = gelu_prime
            elseif activation == "leaky_relu"
                activation = leaky_relu
                activation_prime = leaky_relu_prime
            elseif activation == "swish"
                activation = swish
                activation_prime = swish_prime
            elseif activation == "none_activation"
                activation = none_activation
                activation_prime = none_activation_prime
            else
                @error "activation function not found"
                return 0
            end
            weights = reshape(weights, nw1, nw2)
            bias = reshape(bias, nb1, nb2)
            push!(layers, TurboDense(weights, bias, activation, activation_prime))
        elseif type == "layernorm"
            eps = dic["layer_"*string(ii)*"_eps"]
            push!(layers, TurboLayerNorm(eps))
        elseif type == "norm"
            xmean = reshape(dic["layer_"*string(ii)*"_mean"] .|> Float32, (:, 1))
            xstd = reshape(dic["layer_"*string(ii)*"_std"] .|> Float32, (:, 1))
            eps = dic["layer_"*string(ii)*"_eps"]
            push!(layers, TurboNorm(eps, xmean, xstd))
        else
            @error "Layer Type not found"
        end
    end
    MLP(layers)
end

function convert2turbo(m::Dense)::TurboDense
    weights = m.weight
    swz = weights |> size
    if m.bias == false
        bias = zeros(swz[1], 1)
    else
        bias = reshape(m.bias, (:, 1))
    end
    activation = m.σ |> string
    println("weights: ", weights |> size)
    println("bias: ", bias |> size)
    println("activation: ", activation)
    if activation == "relu"
        activation = relu
        activation_prime = relu_prime
    elseif activation == "gelu"
        activation = gelu
        activation_prime = gelu_prime
    elseif activation == "leakyrelu"
        activation = leaky_relu
        activation_prime = leaky_relu_prime
    elseif activation == "swish"
        activation = swish
        activation_prime = swish_prime
    elseif activation == "none_activation"
        activation = none_activation
        activation_prime = none_activation_prime
    elseif activation == "identity"
        activation = none_activation
        activation_prime = none_activation_prime
    else
        @error "activation function not found"
        return 0
    end
    return TurboDense(weights, bias, activation, activation_prime)
end

function convert2turbo(m::LayerNorm)::TurboLayerNorm
    eps = m.ϵ
    TurboLayerNorm(eps)
end

"""
    convert2turbo(model::Chain)

    convert a flux model to a juliaml model
"""
function convert2turbo(model::Chain)::MLP
    layers = []
    for m in model
        layer = m |> convert2turbo
        push!(layers, layer)
    end
    return MLP(layers)
end

struct ConsModel
    mlpenc::MLP
    mlp::MLP
end

function (m::ConsModel)(x, xtypes)::Matrix{Float32}
    x1 = m.mlpenc(xtypes)
    x2 = vcat(x, x1)
    return m.mlp(x2)
end

function loadconsmodel(fname::String)::Tuple{ConsModel,Vector{String}}
    # add some options here
    mlp = loadmlp(fname * "/mlp.json")
    mlpenc = loadmlp(fname * "/mlpenc.json")
    uniq_ship_types = JSON.parsefile(fname * "/vessel_types_onehot.json"; dicttype=Dict, inttype=Int64, use_mmap=true) .|> String
    ConsModel(mlpenc, mlp), uniq_ship_types
end

function save(model::Chain, fname::String)
    model = convert2turbo(model)
    save(model, fname)
end

@inline function mynormalize(x::Matrix{Float32}, mean::Matrix{Float32}, std::Matrix{Float32}, eps::Float32)::Matrix{Float32}
    @turbo @. (x - mean) / (std + eps)
end

@inline function unmynormalize(x::Matrix{Float32}, mean::Matrix{Float32}, std::Matrix{Float32}, eps::Float32)::Matrix{Float32}
    @turbo @. x * (std + eps) + mean
end

function addnormlayer(mlp::MLP, xmean::Matrix{Float32}, xstd::Matrix{Float32}, eps::Float32)::MLP
    layers = mlp.layers |> deepcopy
    layers = pushfirst!(layers, TurboNorm(eps, xmean, xstd))
    return MLP(layers)
end

# define addition for mlpgradient
function addmlp(a::MLPGradient, b::MLPGradient)::MLPGradient
    layers = []
    for ii in 1:length(a.layers)
        weights = a.layers[ii].weights + b.layers[ii].weights
        bias = a.layers[ii].bias + b.layers[ii].bias
        push!(layers, TurboDenseGradient(weights, bias))
    end
    return MLPGradient(layers)
end

# define + for mlpgradient
function Base.:+(a::MLPGradient, b::MLPGradient)::MLPGradient
    layers = []
    for ii in 1:length(a.layers)
        weights = a.layers[ii].weights + b.layers[ii].weights
        bias = a.layers[ii].bias + b.layers[ii].bias
        push!(layers, TurboDenseGradient(weights, bias))
    end
    return MLPGradient(layers)
end
