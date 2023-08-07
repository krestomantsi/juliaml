module flux

using Flux
using Optimisers
using Plots
using BenchmarkTools

activation = relu
epochs = 100000

x = LinRange(-1, 1, 20)' |> collect .|> Float32
y = sin.(2 * Float32(pi) * x)

model = Chain(Dense(1, 32, activation),
    Dense(32, 32, activation),
    Dense(32, 1))

maeloss(x, y) = Flux.Losses.mae(model(x), y)
mseloss(x, y) = Flux.Losses.mse(model(x), y)

opt = Optimisers.Descent(0.05)
opt_state = Optimisers.setup(opt, model)
params = Flux.params(model)

@time for ii in 1:epochs
    grads, = Flux.gradient((m) -> Flux.Losses.mse(m(x), y), model)
    Optimisers.update!(opt_state, model, grads)
    if ii % 1500 === 0
        println("epoch: ", ii, " |loss: ", mseloss(x, y))
    end
end

x2 = LinRange(-1, 1, 100)' |> collect .|> Float32
y2 = model(x2)
scatter(x', y')
display(plot!(x2', y2'))

display(@benchmark y = model(x))

end # module
