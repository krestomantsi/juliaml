using Plots, XLSX, DataFrames
using BenchmarkTools
using Statistics: mean
using Flux
using Optimisers
using Random
using Interpolations
using JLD2, FileIO

include("julia_utils2.jl")

Plots.theme(:dao)

# setting fixed rng seed
# rng = Random.default_rng()
# Random.seed!(rng, 0)
#mydevice = cpu
mydevice = gpu

#fname = "..\\Vessels\\9574236_InputData.xlsx"
fname = "../Vessels/9574236_InputData.xlsx"
data = DataFrame(XLSX.readtable(fname, "MainInput"))
# %%
Vessel = Dict{String,Any}("g" => 9.80665, "iwave" => 0)
#Vessel["density"] =data[1,"density"]
Vessel["density"] = 1025.0
Vessel["Lpp"] = data[1, "Lpp"] |> float
Vessel["B"] = data[1, "Breadth"] |> float
Vessel["Cb"] = data[1, "Cb"]
Vessel["Kyy"] = data[1, "kyy"]
Vessel["IMO"] = data[1, "IMO"]
Vessel["TS"] = df2fl(data[:, "TS"])
Vessel["S"] = df2fl(data[:, "S"])
Vessel["MCR"] = data[1, "Vessel MCR"]
Vessel["PropD"] = data[1, "PropD"]
Vessel["PropNum"] = data[1, "PropNum"]
Vessel["J"] = df2fl(data[:, "J"])
Vessel["KT"] = df2fl(data[:, "KT"])
Vessel["KQ"] = df2fl(data[:, "KQ"])
Vessel["n0"] = df2fl(data[:, "n0"])
Vessel["Tat"] = df2fl(data[:, "T2"])
Vessel["AT"] = df2fl(data[:, "AT"])
Vessel["nS"] = data[1, "nS"] |> float
Vessel["nGB"] = data[1, "nGB"] |> float
Vessel["Cp"] = data[1, "Cp"]
Vessel["nR"] = data[1, "nR"] |> float
T2 = Vessel["Tat"]
Twind = [minimum(T2), maximum(T2)]
angleB = df2fl(data[:, "angleB"])
angleL = df2fl(data[:, "angleL"])
CxB = df2fl(data[:, "CxB"])
CxL = df2fl(data[:, "CxL"])
CxB2 = interp1d_neqv(angleL, angleB, CxB)
cd1 = vcat(CxB2', CxL')
Vessel["anglescd"] = angleL
Vessel["cdwind"] = cd1
Vessel["Twind"] = Twind
# %% thrust deduction
vs = df2fl(data[:, "vswt"])
t = df2fl(data[:, "twt"])
x = hcat(t, vs)' |> collect
y = df2fl(data[:, "thdf"])' |> collect

T = df2fl(data[:, "T"])
ts = unique(T)
T = vcat(T, ts)
Vs = df2fl(data[:, "Vs"])
Vs = vcat(Vs, 0 * ts)
Pwr = df2fl(data[:, "Pwr"])
Pwr = vcat(Pwr, 0 * ts)
X = hcat(T, Vs)' |> collect
xmean = mean(X, dims=2)
xstd = std(X, dims=2)
tvmean = xmean
tvstd = xstd

#acti = swish
acti = relu
wd = 1e-3
lr = 1e-3
epochs = 30000
lat = 8


# thrust deduction factor
# xmean = mean(x, dims=2)
# xstd = std(x, dims=2)
# xmax = maximum(x, dims=2)
# xmin = minimum(x, dims=2)

function mynormalize(x, xmean, xstd)
    x = @. (x - xmean) / (xstd .+ 1.0f-8)
    return x
end

function standardize(x, xmin, xmax)
    x = (x .- xmin) ./ (xmax .- xmin .+ 1e-8)
    return x
end

x = mynormalize(x, xmean, xstd)
#x = standardize(x, xmin, xmax)

opt = Flux.AdamW(lr, (0.9, 0.999), wd)

function mse_loss(model, data)
    y_pred = model(data[1])
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss
end

function trainingmy(mlp::Chain, st_opt, epochs, tdata)::Chain
    @time for ii in 1:epochs
        grads = Flux.gradient(m -> mse_loss(m, tdata), mlp)[1]
        Flux.update!(st_opt, mlp, grads)
        if ii % 5000 == 0
            println("error ", mse_loss(mlp, tdata))
        end
    end
    return mlp
end

thlista = [lat, lat]
thmlp = Chain(
    Dense(2, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

tdata = (x, y)
st_opt = Flux.setup(opt, thmlp)
thmlp = trainingmy(thmlp, st_opt, epochs, tdata)

scatter(vs, y', label="Values", legend=:topleft, marker_z=t)
for ii in 1:4
    global t, tu, tt, vss, tm, xx, yy, x, st
    tu = unique(t)
    tt = collect(LinRange(minimum(tu), maximum(tu), 4))
    vss = collect(LinRange(0, 26, 20))
    tm = tt[ii]
    #println(tm)
    x = hcat(tm .+ 0 .* vss, vss)'
    x = mynormalize(x, xmean, xstd)

    yy = thmlp(x)
    plot!(vss, yy', label="Draft: " * string(tm) * " [m]", legend=:topleft)
end
title!("thrust deduction")
xlabel!("Speed [knots]")
savefig("thrust.png")

# %% wake fraction
vs = df2fl(data[:, "vswt"])
t = df2fl(data[:, "twt"])
x = hcat(t, vs)' |> collect
y = df2fl(data[:, "wfts"])' |> collect

x = mynormalize(x, xmean, xstd)

lr = 1e-3
wd = 1e-3
lat = 8
wlist = [lat, lat]
wmlp = Chain(
    Dense(2, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

opt = Flux.AdamW(lr, (0.9, 0.999), wd)
st_opt = Flux.setup(opt, wmlp)
tdata = (x, y)
wmlp = trainingmy(wmlp, st_opt, epochs, tdata)

scatter(vs, y', label="Values", legend=:topleft, marker_z=t)
for ii in 1:4
    global t, tu, tt, vss, tm, xx, yy, x, st
    tu = unique(t)
    tt = collect(LinRange(minimum(tu), maximum(tu), 4))
    vss = collect(LinRange(0, 26, 20))
    tm = tt[ii]
    #println(tm)
    x = hcat(tm .+ 0 .* vss, vss)' |> collect
    x = mynormalize(x, xmean, xstd)
    yy = wmlp(x)
    plot!(vss, yy', label="Draft: " * string(tm) * " [m]", legend=:topleft)
end
title!("wake factor")
xlabel!("Speed [knots]")
savefig("wake.png")

# # %% rpmparams
vs = df2fl(data[:, "vs3"])
t = df2fl(data[:, "T3"])
x = hcat(t, vs)' |> collect
y = df2fl(data[:, "rpm"])' |> collect

x = mynormalize(x, xmean, xstd)

lr = 1e-2
wd = 1e-2
lat = 32
epochs = 50000
rpmlist = [lat, lat]
rpmmlp = Chain(
    Dense(2, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

opt = Flux.AdamW(lr, (0.9, 0.999), wd)
st_opt = Flux.setup(opt, rpmmlp)
tdata = (x, y)
rpmmlp = trainingmy(rpmmlp, st_opt, epochs, tdata)

scatter(vs, y', label="Values", legend=:topleft, marker_z=t)
for ii in 1:4
    global t, tu, tt, vss, tm, xx, yy, x, st
    tu = unique(t)
    tt = collect(LinRange(minimum(tu), maximum(tu), 4))
    vss = collect(LinRange(10, 19, 20))
    tm = tt[ii]
    x = hcat(tm .+ 0 .* vss, vss)' |> collect
    x = mynormalize(x, xmean, xstd)
    yy = rpmmlp(x)
    plot!(vss, yy', label="Draft: " * string(tm) * " [m]", legend=:topleft)
end
title!("speed vs rpm")
xlabel!("Speed [knots]")
savefig("rpm.png")

# ct training
T = df2fl(data[:, "T"])
ts = unique(T)
T = vcat(T, ts)
Vs = df2fl(data[:, "Vs"])
Vs = vcat(Vs, 0 * ts)
Pwr = df2fl(data[:, "Pwr"])
Pwr = vcat(Pwr, 0 * ts)
X = hcat(T, Vs)' |> collect
X = mynormalize(X, xmean, xstd)

rpmv = rpmmlp(X)'
tv = thmlp(X)'
wv = wmlp(X)'
nH = (1.0 .- tv) ./ (1.0 .- wv)
D = Vessel["PropD"]
Jv = 0.5144444 * (1.0 .- wv) .* Vs ./ (rpmv ./ 60.0 * D)
n0v = interp1d_neqv(Jv, Vessel["J"], Vessel["n0"])
nR = Vessel["nR"]
nGB = Vessel["nGB"]
nS = Vessel["nS"]
#Rcalm = Pwr .* (nGB * nS * dropdims(n0v, dims=2) .* nH * nR) ./ (Vs * 0.5144444)
Rcalm = Pwr .* (nGB * nS * n0v .* nH * nR) ./ (Vs * 0.5144444)
Rcalm[isnan.(Rcalm), 1] .= 0
Ts = Vessel["TS"]
S = Vessel["S"]
ss = interp1d_neqv(T, Ts, S)
CT = Rcalm ./ ((0.5144444 * Vs) .^ 2 .* ss / 1000 * Vessel["density"] / 1000)
lmao = minimum(CT[.!isnan.(CT)])
CT[isnan.(CT)] .= lmao
#CT[isnan.(CT)] .= 0
Y = CT'
x = X |> collect # has already been normalized
y = Y |> collect


epochs = 30000
lr = 1e-2
wd = 1e-4
lat = 16
acti = swish
ctlist = [lat, lat]
ctmlp = Chain(
    Dense(2, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

opt = Flux.AdamW(lr, (0.9, 0.999), wd)
#opt = Flux.Adam(lr)
st_opt = Flux.setup(opt, ctmlp)
tdata = (x, y)
ctmlp = trainingmy(ctmlp, st_opt, epochs, tdata)

scatter(Vs, Y', label="values", legend=:topleft, marker_z=T)
for ii in 1:4
    global t, tu, tt, vss, tm, xx, yy, x, st
    tu = unique(T)
    tt = collect(LinRange(minimum(tu), maximum(tu), 4))
    vss = collect(LinRange(0.0, 25, 40))
    #vss = collect(LinRange(13,19,20))
    tm = tt[ii]
    x = hcat(tm .+ 0 .* vss, vss)' |> collect
    x = mynormalize(x, xmean, xstd)
    yy = ctmlp(x)
    plot!(vss, yy', label="Draft: " * string(tm) * " [m]", legend=:topleft)
end
title!("speed-CT")
xlims!(0.0, 25)
savefig("CT.png")
# %% fuel
x = df2fl(data[:, "PropPower"])' |> collect
y = df2fl(data[:, "Fuel1"])' |> collect
xmean = mean(x, dims=2)
xstd = std(x, dims=2)
pmean = xmean
pstd = xstd
x = mynormalize(x, xmean, xstd)

lr = 1e-2
wd = 1e-3
lat = 16
acti = Flux.swish
sfoclist = [lat, lat]
sfocmlp = Chain(
    Dense(1, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

opt = Flux.AdamW(lr, (0.9, 0.999), wd)
st_opt = Flux.setup(opt, sfocmlp)
tdata = (x, y)
sfocmlp = trainingmy(sfocmlp, st_opt, epochs, tdata)

scatter(x', y', label="", legend=:topleft)
x2 = collect(LinRange(0, Vessel["MCR"] * 1.2, 50))'
x2 = mynormalize(x2, xmean, xstd)
y2 = sfocmlp(x2)
plot!(x2', y2', label="", legend=:topleft)
xlabel!("M/E Power [kw]")
ylabel!("Consumption [gr/kWh]")
title!("M/E Specific Fuel Consumption (ISO)")
savefig("sfoc.png")
# added wave part
vs = LinRange(0.01, 25, 15)
hs = LinRange(0.01, 8, 15)
tm = LinRange(9, 21, 4)
tp = LinRange(0.01, 18, 10)

# speed , height, draft , period
function meshgridStaw(vs, hs, tm, tp)
    s1, s2, s3, s4 = [size(x)[1] for x in (vs, hs, tm, tp)]
    x = zeros(4, s1 * s2 * s3 * s4)
    y = zeros(1, s1 * s2 * s3 * s4)
    ind = 1
    w, weights = gausslegendre(200)
    for vsi in vs
        for hsi in hs
            for tmi in tm
                for tpi in tp
                    x[:, ind] .= [vsi, hsi, tpi, tmi]
                    y[1, ind] = StawaveD2(vsi, hsi, tpi, tmi, Vessel, w, weights)
                    ind += 1
                end
            end
        end
    end
    return x, y
end
x, y = meshgridStaw(vs, hs, tm, tp)

xmean = mean(x, dims=2)
xstd = std(x, dims=2)
d2mean = xmean
d2std = xstd
# non dimentionalize y
# froude
fr = x[1, :] / sqrt(9.81 * Vessel["Lpp"])
# wave steepness
s = (2 * pi) * x[2, :] ./ (x[4, :] .^ 2 * 9.81)

y = y ./ (Vessel["Lpp"] * x[2, :])'

x = mynormalize(x, xmean, xstd)

lr = 1e-2
wd = 1e-3
lat = 32
epochs = 50000
acti = Flux.swish
d2list = [lat, lat]
d2mlp = Chain(
    Dense(4, d2list[1], acti),
    Dense(d2list[1], d2list[2], acti),
    LayerNorm(d2list[2]),
    Dense(d2list[2], 1))


# opt = Flux.AdamW(lr, (0.9, 0.999), wd)
opt = Flux.Adam(lr)
st_opt = Flux.setup(opt, d2mlp)
x = x .|> Float32
y = y .|> Float32
tdata = (x, y) |> mydevice

d2mlpgpu = d2mlp |> mydevice
tdatagpu = tdata |> mydevice
st_optgpu = st_opt |> mydevice
d2mlp = trainingmy(d2mlpgpu, st_optgpu, epochs, tdata) |> cpu

vs2 = LinRange(0, 20, 100)
tm2 = 15.0:15.0
tp2 = 9:9
#hs2 = 1.00:1.00
hs2 = 0.50:0.50
x2, y2 = meshgridStaw(vs2, hs2, tm2, tp2)
hs2 = x2[2, :]
tm2 = x2[3, :]
x2 = mynormalize(x2, xmean, xstd)
y22 = d2mlp(x2)
y22 = y22 .* (Vessel["Lpp"] * hs2)'

scatter(collect(vs2), y2', label="values")
plot!(collect(vs2), y22', label="model")
title!("Added wave ")
savefig("D2mpl.png")
# benchmark d2 gradients
# @benchmark Flux.gradient((m) -> sum(m(x2)), d2mlp)
# ofc due to scaling reasons 64 is more than x4 times slower than 32
# faster anglecor
th = LinRange(0.0, 2 * pi, 200)' |> collect
yth = SpAngleCorr.(th)
x = th
y = yth

lr = 1e-2
wd = 1e-5
lat = 16
epochs = 50000
acti = Flux.swish
spanglelist = [lat, lat]
spanglemlp = Chain(
    Dense(1, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

#opt = Flux.AdamW(lr, (0.9, 0.999), wd)
opt = Flux.Adam(lr)
st_opt = Flux.setup(opt, spanglemlp)
tdata = (x, y)
spanglemlp = trainingmy(spanglemlp, st_opt, epochs, tdata)
yth2 = spanglemlp(th)
scatter(th', yth', label="og")
plot!(th', yth2', color="red", label="poly")
savefig("delete.png")
# reverse j solver
cv = LinRange(0.0, 10, 200)' |> collect
yj = map(ci -> NewtonRaphsonJ_forloop(ci, 0.5, Vessel["J"], Vessel["KT"])[1], cv)
x = cv
y = yj

lr = 1e-2
wd = 1e-5
lat = 8
epochs = 30000
acti = Flux.swish
j2list = [lat, lat]
j2mlp = Chain(
    Dense(1, lat, acti),
    Dense(lat, lat, acti),
    Dense(lat, 1))

#opt = Flux.AdamW(lr, (0.9, 0.999), wd)
opt = Flux.Adam(lr)
st_opt = Flux.setup(opt, j2mlp)
tdata = (x, y)
j2mlp = trainingmy(j2mlp, st_opt, epochs, tdata)

yj2 = j2mlp(cv)
scatter(cv', yj', label="og reverse solver")
plot!(cv', yj2, label="poly model")

th2 = LinRange(0, 20, 200)' |> collect
yy = j2mlp(th2)
scatter(cv', yj', label="og reverse solver")
plot!(th2', yy')
savefig("reversesolver.png")
# creating a Vessel struct to store everything and do perf calculations
# cast to float32
tvmean = tvmean .|> Float32
tvstd = tvstd .|> Float32
pmean = pmean .|> Float32
pstd = pstd .|> Float32
d2mean = d2mean .|> Float32
d2std = d2std .|> Float32

Vessel = fmap((x) -> Float32.(x), Vessel) # makes all matrixes float32

cdi = interpolate((Vessel["Twind"], Vessel["anglescd"]), Vessel["cdwind"], Gridded(Linear()))
Si = linear_interpolation(Vessel["TS"], Vessel["S"])
Tati = linear_interpolation(Vessel["Tat"], Vessel["AT"])
n0i = linear_interpolation(Vessel["J"], Vessel["n0"])

Vessel0 = MyVessel(Vessel["MCR"], Vessel["Lpp"], Vessel["B"], wmlp, thmlp, ctmlp, d2mlp, j2mlp, sfocmlp, spanglemlp, tvmean, tvstd, pmean, pstd, d2mean, d2std, Vessel["J"], Vessel["AT"], Vessel["Tat"], Vessel["Twind"], Vessel["cdwind"], Vessel["anglescd"], Vessel["PropNum"], Vessel["density"], Vessel["PropD"], Vessel["n0"], Vessel["nR"], Vessel["nGB"], Vessel["nS"], Vessel["TS"], Vessel["S"], cdi, Si, Tati, n0i)
Vesseli = MyVesseli(Vessel["MCR"], Vessel["Lpp"], Vessel["B"], wmlp, thmlp, ctmlp, d2mlp, j2mlp, sfocmlp, spanglemlp, tvmean, tvstd, pmean, pstd, d2mean, d2std, Vessel["J"], Vessel["AT"], Vessel["Tat"], Vessel["Twind"], Vessel["cdwind"], Vessel["anglescd"], Vessel["PropNum"], Vessel["density"], Vessel["PropD"], Vessel["n0"], Vessel["nR"], Vessel["nGB"], Vessel["nS"], Vessel["TS"], Vessel["S"])
model_state = Flux.state(Vesseli)


IMO = data[1, "IMO"]
fname2 = string("../Vessels/Vessel_", string(IMO)) * ".jld2"
fname3 = string("../Vessels/Vessel_", string(IMO)) * "_struct" * ".jld2"
JLD2.save(fname3, "Vessel", Vessel0)
jldsave(fname2; model_state)
# make every matrix in vessel float 32
#fmap(x -> Float32.(x), Vessel)
# testing performance
ms = JLD2.load(fname2, "model_state")
# initiate a vesseli struct with dummy values (do not use any existing values and or mlp)
Vessel0 = JLD2.load(fname3, "Vessel")
Vesseli = MyVesseli(ms.MCR, ms.Lpp, ms.B, Vessel0.wmlp, Vessel0.thmlp, Vessel0.ctmlp, Vessel0.d2mlp, Vessel0.j2mlp, Vessel0.sfocmlp, Vessel0.spanglemlp, ms.xmean, ms.xstd, ms.pmean, ms.pstd, ms.d2mean, ms.d2std, ms.J, ms.at, ms.tat, ms.Twind, ms.cdwind, ms.anglescd, ms.PropNum, ms.waterdensity, ms.D, ms.n0, ms.nR, ms.nGB, ms.nS, ms.TS, ms.S)
Vesseli = Flux.loadmodel!(Vesseli, ms)
cdi = interpolate((Vesseli.Twind, Vesseli.anglescd), Vesseli.cdwind, Gridded(Linear()))
Si = linear_interpolation(Vesseli.TS, Vesseli.S)
Tati = linear_interpolation(Vesseli.tat, Vesseli.at)
n0i = linear_interpolation(Vesseli.J, Vesseli.n0)
Vessel0 = MyVessel(Vesseli.MCR, Vesseli.Lpp, Vesseli.B, Vesseli.wmlp, Vesseli.thmlp, Vesseli.ctmlp, Vesseli.d2mlp, Vesseli.j2mlp, Vesseli.sfocmlp, Vesseli.spanglemlp, Vesseli.xmean, Vesseli.xstd, Vesseli.pmean, Vesseli.pstd, Vesseli.d2mean, Vesseli.d2std, Vesseli.J, Vesseli.at, Vesseli.tat, Vesseli.Twind, Vesseli.cdwind, Vesseli.anglescd, Vesseli.PropNum, Vesseli.waterdensity, Vesseli.D, Vesseli.n0, Vesseli.nR, Vesseli.nGB, Vesseli.nS, Vesseli.TS, Vesseli.S, cdi, Si, Tati, n0i)


dumpa = zeros(Float32, 100)
IMO = 9574236
Vs = 15.0f0 .+ dumpa
course = 0.0f0 .+ dumpa
Tm = 15.0f0 .+ dumpa
Hs = 1.5f0 .+ dumpa
Tp = 7.0f0 .+ dumpa
wavedir = 180.0f0 .+ dumpa
Vw = 11.0f0 .+ dumpa
winddir = 180.0f0 .+ dumpa
fouling_cf = 1.0f0
power_cf = 1.0f0
# time of truth benchmarking performance
# inference performance
display(@benchmark Performance2(Vs, course, Tm, Hs, Tp, wavedir, Vw, winddir, Vessel0, fouling_cf, power_cf))
# backprop performance
display(@benchmark Flux.gradient((Vs, course, Tm, Hs, Tp, wavedir, Vw, winddir) -> sum(Performance2(Vs, course, Tm, Hs, Tp, wavedir, Vw, winddir, Vessel0, power_cf, fouling_cf)), Vs, course, Tm, Hs, Tp, wavedir, Vw, winddir))
# ok that was bad
# culprit number 1 9ms out of 14 ms
#display(@benchmark Flux.gradient((course, Tm, Vs, Vw, winddir) -> sum(map((course, Tm, Vs, Vw, winddir) -> Wind_Force2(course, Tm, Vs, Vw, winddir, Vessel0), course, Tm, Vs, Vw, winddir)'), course, Tm, Vs, Vw, winddir))
#
# # gradient of interp1d
# display(@benchmark Flux.gradient((x) -> sum(map((x) -> interp1d_neq(x, Vessel0.tat, Vessel0.at), x)), Vs))
#
# # gradient of interp2d
# display(@benchmark Flux.gradient((x, y) -> sum(map((x, y) -> interp2d_neq(x, y, Vessel0.Twind, Vessel0.anglescd, Vessel0.cdwind), x, y)), Vs, Vs))
#
# display(@benchmark sum(map((x, y) -> interp2d_neq(x, y, Vessel0.Twind, Vessel0.anglescd, Vessel0.cdwind), Vs, Vs)))
#
# display(@benchmark Flux.gradient((x, y) -> interp2d_neq(x, y, Vessel0.Twind, Vessel0.anglescd, Vessel0.cdwind), 0.5f0, 0.4f0))
#
# cdi = interpolate((Vessel0.Twind, Vessel0.anglescd), Vessel0.cdwind, Gridded(Linear()))
#
# display(@benchmark sum(map(cdi, Vs, Vs)))
#
# display(@benchmark Flux.gradient((x, y) -> sum(map((x, y) -> cdi(x, y), x, y)), Vs, Vs))
