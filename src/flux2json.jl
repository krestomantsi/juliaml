# a script to convert flux mlp params into a json readable file
using Flux
using Plots
using Optimisers
using BenchmarkTools
using JSON2

activation = relu
epochs = 30000
lr = 1e-2

x = LinRange(-1, 1, 20)' |> collect .|> Float32
y = x .^ 2

model = Chain(Dense(1, 32, activation),
    Dense(32, 32, activation),
    Dense(32, 1))

opt = Optimisers.Adam(lr)
opt_state = Optimisers.setup(opt, model)
params = Flux.params(model)

maeloss(x, y) = Flux.Losses.mae(model(x), y)
mseloss(x, y) = Flux.Losses.mse(model(x), y)

@time for ii in 1:epochs
    grads, = gradient((m) -> Flux.Losses.mse(m(x), y), model)
    Optimisers.update!(opt_state, model, grads)
    if ii % 1500 == 0
        println("epoch: ", ii, " |loss: ", maeloss(x, y))
    end
end


x2 = LinRange(-1.2, 1.2, 100)' |> collect .|> Float32
y2 = model(x2)
scatter(x', y')
display(plot!(x2', y2'))


params = Flux.params(model)
