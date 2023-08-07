using Flux
using Plots
using Statistics: mean

x = LinRange(-1.0, 1, 20)' |> collect .|> Float32
y = sin.(2 * pi * x) .|> Float32
latent = 16
latent2 = 32
acti = relu
#y = x .^ 2


model = Chain(
    Dense(1, latent, acti, init=ones),
    Dense(latent, latent2, acti, init=ones),
    Dense(latent2, 1, init=ones))


function myloss(m, x, y)
    mean((m(x) .- y) .^ 2)
end
grads = Flux.gradient((m) -> myloss(m, x, y), model)[1]

ii = 2
display(grads.layers[ii].weight)
display(grads.layers[ii].bias)

model(x)
