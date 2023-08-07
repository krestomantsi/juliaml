# module juliaml

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
hidden_size = 32
activation = relu
activation_prime = relu_prime
epochs = 100000
lr = 0.05f0
n = 20

model = MLP(input_size, hidden_size, output_size, activation, activation_prime)

#x = randn(Float32, input_size, 1000)
x = LinRange(-1, 1, n)' |> collect .|> Float32

y = sin.(2 * Float32(pi) * x)
# y = x .^ 2
y2 = model(x)

# println("Inference benchmark")
display(@benchmark y2 = model(x))

# @report_opt model(x)
# @report_opt model.layers[1](x)


pullback, grads = backward(model.layers[1], rand(Float32, input_size, 10) |> collect, ones(Float32, 32, 10) |> collect)
outputs, grads = backward(model, x, y, mse_prime)

# display(@benchmark outputs, grads = backward(model, x, y, mse_prime))

for ii in 1:3
    println("Layer ", ii, "------------------------------------")
    local outputs, grads = backward(model, x, y, mse_prime)
    println("Layer ", ii, " output size", outputs[ii+1] |> size)
    println("Layer ", ii, " model weights size", model.layers[ii].weights |> size)
    println("Layer ", ii, " model bias size", model.layers[ii].bias |> size)
    println("Layer ", ii, " grads weights size", grads.layers[ii].weights |> size)
    println("Layer ", ii, " grads bias size", grads.layers[ii].bias |> size)
end


# outputs, grads = backward(model, x, y, mse_prime)
# sgd!(model, grads, lr)
# model.layers[2].weights


@time model = trainsgd(model, x, y, mse, mse_prime, epochs, lr)
# excuse me julia what is this??
# everytime i run the same script in the same file
# the training gets slower and slower
# possible memory leak?? thanks julia!

displaynetwork(model, x, y, mse_prime)

x2 = LinRange(-1, 1, 200)' |> collect .|> Float32
y2 = model(x2)

scatter(x', y', label="data")
plot!(x2', y2', label="model")
savefig("result.png")



# lmao flux why are u so bad
# i deleted the flux benchmarks but if u want try urselves
# josh doesnt know multivariate calc LMAOO
# end # module juliaml

x2 = LinRange(-5, 5, 100)' |> collect .|> Float32
y2 = gelu_prime(x2)
plot(x2', y2')
