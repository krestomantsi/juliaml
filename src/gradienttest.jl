using Flux
using Plots
using Statistics: mean

x = LinRange(-1.0, 1, 10)' |> collect .|> Float32
y = sin.(2 * pi * x) .|> Float32
#y = x .^ 2

model = Chain(
    Dense(1, 8, relu, init=ones),
    Dense(8, 8, relu, init=ones),
    Dense(8, 1, init=ones))


function myloss(m, x, y)
    mean((m(x) .- y) .^ 2)
end
grads = Flux.gradient((m) -> myloss(m, x, y), model)[1]

ii = 3
display(grads.layers[ii].weight)
display(grads.layers[ii].bias)
