module juliaml

using LinearAlgebra
using LoopVectorization
using Plots
using Random
using Statistics: mean, std
using JET
using Base.Threads: Threads, @spawn
using DataFrames
using CSV
using JSON
using PrecompileTools
using Flux
using Folds

include("utils.jl")

export MLP, MLPGradient, train!, displaynetwork, save, loadmlp, swish, swish_prime, relu, relu_prime, mse, mse_prime, TurboNorm, TurboDense, TurboLayerNorm
export gelu, gelu_prime, none_activation, none_activation_prime, convert2turbo, mynormalize, addnormlayer, unmynormalize, backward
# delete those exports
# export addmlp, chunk

@compile_workload begin
    input_size = 1
    output_size = 1
    hidden_size = 32
    activation = swish
    activation_prime = swish_prime
    epochs = 10
    lr = 0.01f0
    wd = 0.00001f0
    n = 10

    model = MLP(input_size, hidden_size, output_size, activation, activation_prime)

    x = LinRange(-1, 1, n)' |> collect .|> Float32

    y = sin.(4 * Float32(pi) * x)
    #y = cos.(3 * Float32(pi) * x) .^ 11

    y2 = model(x)

    # println("Inference benchmark")
    #display(@benchmark y2 = model(x))

    # @report_opt model(x)
    # @report_opt model.layers[1](x)

    # pullback, grads = backward(model.layers[1], rand(Float32, input_size, 10) |> collect, ones(Float32, 32, 10) |> collect)
    outputs, grads = backward(model, x, y, mse_prime)

    # display(@benchmark outputs, grads = backward(model, x, y, mse_prime))

    model = train!(model, x, y, lr, wd, epochs, mse, mse_prime, false)

    # for ii in 1:3
    #     println("Layer ", ii, "------------------------------------")
    #     local outputs, grads = backward(model, x, y, mse_prime)
    #     println("Layer ", ii, " output size", outputs[ii+1] |> size)
    #     println("Layer ", ii, " model weights size", model.layers[ii].weights |> size)
    #     println("Layer ", ii, " model bias size", model.layers[ii].bias |> size)
    #     println("Layer ", ii, " grads weights size", grads.layers[ii].weights |> size)
    #     println("Layer ", ii, " grads bias size", grads.layers[ii].bias |> size)
    # end

    # displaynetwork(model, x, y, mse_prime)

    # testing saving & loading
    save(model, joinpath(@__DIR__, "model.json"))
    model2 = loadmlp(joinpath(@__DIR__, "model.json"))
end # end of @compile_workload

end
