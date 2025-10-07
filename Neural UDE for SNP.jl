using JLD, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, OptimizationOptimisers
using ComponentArrays, StableRNGs, LineSearches, LinearAlgebra, Statistics, CSV, DataFrames

#This code is for the last case of 50 training points and 10 forecasting points. For other cases, only the starred
#commented sections need to be altered by changing the indexation.  

# load and pre-process the data 
df = CSV.read("C:/Users/ISHAAN/Downloads/SNP FINAL.csv", DataFrame)

# *Select training range (50 days) 
Prices_raw = log.(df.Price[2455:2455+49])  
Drifts_raw = df.log_return[2455:2455+49]

# Store mean & standard deviation 
μ_P, σ_P = mean(Prices_raw), std(Prices_raw)
μ_D, σ_D = mean(Drifts_raw), std(Drifts_raw)

# Normalize training series
Prices = (Prices_raw .- μ_P) ./ σ_P
Drifts = (Drifts_raw .- μ_D) ./ σ_D

# *Full window of 60 points
Prices_all_raw = log.(df.Price[2455:2455+59])  
Drifts_all_raw  = df.log_return[2455:2455+59]

# Normalize full window using training stats (so comparisons are consistent)
Prices_all = (Prices_all_raw .- μ_P) ./ σ_P
Drifts_all = (Drifts_all_raw .- μ_D) ./ σ_D

# Neural network setup
rng = StableRNG(1234)
ann = Lux.Chain(Lux.Dense(2, 14, tanh), Lux.Dense(14, 1))
p1, st1 = Lux.setup(rng, ann)
p1 = ComponentArray(p1) 
parameter_array = Float64[0.0023700149450021205 0.0138] #*paramters will change according to training period
p0_vec = ComponentArray(layer_1 = p1, layer_2 = parameter_array)

# UDE
function SUNM(du, u, p, t)
    (S, U) = u
    alpha = abs(p.layer_2[2])
    mubar = abs(p.layer_2[1])
    UDE_term = ann([u[1]; t], p.layer_1, st1)[1][1]
    du[1] = u[2]                     
    du[2] = -alpha * (u[2] - mubar) + UDE_term
end

# *Problem setup (training period)
u0 = Float64[Prices[1], Drifts[1]]
tspan = (0.0, 50.0)
datasize = 50
t = range(tspan[1], tspan[2], length=datasize)
prob = ODEProblem(SUNM, u0, tspan, p0_vec)

# Prediction & Loss
function predict_adjoint(θ)
    Array(solve(prob, AutoTsit5(Rosenbrock23()), p=θ, saveat=t, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss_adjoint(θ)
    x = predict_adjoint(θ)
    w1, w2 = 1.0, std(Prices) / std(Drifts)

    # Data-fitting loss
    l_data = w1 * sum(abs2, Prices .- x[1, :]) +
         w2 * sum(abs2, Drifts .- x[2, :])

    # L2 regularization term (penalize large NN weights)
    λ = 1e-4  # regularization strength
    l_l2 = λ * sum(abs2, ComponentArrays.getdata(θ.layer_1))

    return l_data + l_l2
end

# Optimization
iter = 0
function callback2(θ, l)
    global iter
    iter += 1
    if iter % 100 == 0
        println("Iteration $iter, Loss: $l")
    end
    return l > 1e6
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p0_vec)
res1 = Optimization.solve(optprob, ADAM(0.1), callback=callback2, maxiters=1000)
optprob_bfgs = remake(optprob; u0=res1.u)
res2 = Optimization.solve(optprob_bfgs, BFGS(initial_stepnorm=0.1), callback=callback2, maxiters=1000)

# Training prediction
data_pred = predict_adjoint(res2.u)

# *Forecast for next 10 days, give the total predict+forecast period here, and the code will forecast for next 10 days
tspan_forecast = (0.0, 60.0)  
t_forecast = range(tspan_forecast[1], tspan_forecast[2], length=datasize + 10)

prob_forecast = remake(prob; tspan=tspan_forecast)
forecast_pred = Array(solve(prob_forecast, AutoTsit5(Rosenbrock23()), 
                             p=res2.u, saveat=t_forecast,
                             sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))

train_pred = forecast_pred[:, 1:datasize]
future_pred = forecast_pred[:, datasize+1:end]

# Forecast evaluation metrics

# Actual future values (normalized)
Prices_future_actual = Prices_all[datasize+1:end]
Drifts_future_actual = Drifts_all[datasize+1:end]

# RMSE for forecast prices and drifts
rmse_price = sqrt(mean((future_pred[1, :] .- Prices_future_actual).^2))
rmse_drift = sqrt(mean((future_pred[2, :] .- Drifts_future_actual).^2))

println("Forecast RMSE (Price): ", rmse_price)
println("Forecast RMSE (Drift): ", rmse_drift)

# Directional hit rate (percentage of days predicted in correct direction)
price_actual_diff = diff(Prices_future_actual)
price_pred_diff = diff(future_pred[1, :])
hit_rate_price = mean(sign.(price_actual_diff) .== sign.(price_pred_diff)) * 100

drift_actual_diff = diff(Drifts_future_actual)
drift_pred_diff = diff(future_pred[2, :])
hit_rate_drift = mean(sign.(drift_actual_diff) .== sign.(drift_pred_diff)) * 100

println("Directional Hit Rate (Price): ", hit_rate_price, "%")
println("Directional Hit Rate (Drift): ", hit_rate_drift, "%")

#Closed-form verification (Nθ ≡ 0) 

# Parameters from trained model (note: we used normalized log-prices & drifts)
α_est   = abs(res2.u.layer_2[2])
μbar_est = abs(res2.u.layer_2[1])
μ0_est = Drifts[1]
S0_est = Prices[1]

# Time grid matching forecast_pred 
times = collect(t_forecast .- t_forecast[1])   # start at 0

# Closed-form solution for log-price S(t) when Nθ ≡ 0:
# μ(t) = μbar + (μ0 - μbar) * exp(-α t)
# logS(t) = S0 + μbar * t + (μ0 - μbar)/α * (1 - exp(-α t))
function closedform_logS(α, μbar, μ0, S0, times)
    μt = μbar .+ (μ0 .- μbar) .* exp.(-α .* times)              
    integral = μbar .* times .+ ((μ0 .- μbar) ./ α) .* (1 .- exp.(-α .* times))
    return S0 .+ integral, μt
end

S_closed, μ_closed = closedform_logS(α_est, μbar_est, μ0_est, S0_est, times)

# Now solve numerically with NN forced to zero by temporarily replacing SUNM with zero-NN version
function SUNM_zero(du, u, p, t)
    (S, U) = u
    alpha = abs(p.layer_2[2])
    mubar = abs(p.layer_2[1])
    UDE_term = 0.0
    du[1] = u[2]
    du[2] = -alpha * (u[2] - mubar) + UDE_term
end

prob_zero = ODEProblem(SUNM_zero, u0, (times[1], times[end]), res2.u)  # use trained α, μbar
sol_zero = Array(solve(prob_zero, AutoTsit5(Rosenbrock23()), saveat=times))

S_zero = sol_zero[1, :] 

# Compare closed form vs numerical zero-NN (PRICE)
closed_vs_num_rmse = sqrt(mean((S_closed .- S_zero).^2))
println("Closed-form vs numerical (Nθ=0) RMSE on log-price: ", closed_vs_num_rmse)
println("Max abs diff: ", maximum(abs.(S_closed .- S_zero)))

# Compare closed form vs numerical zero-NN (DRIFT)
U_zero = sol_zero[2, :]   # numerical drift from zero-NN solve
closed_vs_num_rmse_drift = sqrt(mean((μ_closed .- U_zero).^2))
println("Closed-form vs numerical (Nθ=0) RMSE on drift: ", closed_vs_num_rmse_drift)
println("Max abs diff (drift): ", maximum(abs.(μ_closed .- U_zero)))

# Plot results with actuals for forecast period

# *Get actual prices & drifts for full forecast period (train + future)
Prices_all = log.(df.Price[2455:2455+59])
Drifts_all = df.log_return[2455:2455+59]

# Normalize using training stats
Prices_all = (Prices_all .- μ_P) ./ σ_P
Drifts_all = (Drifts_all .- μ_D) ./ σ_D

# Plot Prices
plot(t_forecast, Prices_all, label="Actual Price", color="red", lw=2, 
     xlabel="Time", ylabel="Normalized Value of S&P", 
     title="S&P Actual vs. Predicted Values", 
     titlefontsize=11)

plot!(t_forecast, forecast_pred[1, :], label="Predicted+Forecasted Price", color="orange", lw=2, ls=:dash)

# Plot Drifts
plot!(t_forecast, Drifts_all, label="Actual Drift", color="blue", lw=2)
plot!(t_forecast, forecast_pred[2, :], label="Predicted+Forecasted Drift", color="cyan", lw=2, ls=:dash)

# *Add vertical line at boundary
vline!([50.0], label="Forecast Start", color=:black, lw=2, ls=:dot)