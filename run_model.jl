#!/usr/bin/env julia
using Serialization

include("stochastic_employment.jl")

# Set parameters
β = 0.9f0

δ = 0.1f0
α  = 1f0
η = 0f0
A = 1.5f0

γ = 1f0
θ = 1.5f0
ε = 0.5f0

z = 1f0
r = 0.1f0

λ_f = 0.95f0
λ_l = 0.05f0

# Set up grid lengths
h_N = 200
a_N = 200
e_N = 2

# Define parameters
a_lbar = -z / r * 0.99f0
h_bar = (A /  δ)^(1 / (1-η))

# Set up grids
h_grid = collect(Float32, range(z, h_bar, h_N))
a_grid = collect(Float32, range(a_lbar, -a_lbar, a_N))
e_grid, trans_mat = calculate_emp_process(λ_f, λ_l)

if !isinteractive()
    results = do_VFI(
        flow_value, h_grid, a_grid, e_grid, trans_mat, β,
        δ=δ, α=α, η=η, A=A, γ=γ, θ=θ, ε=ε, z=z, r=r, tol=1f-6
    )

    open(joinpath(@__DIR__, "results", "VFI_results.jls"), "w") do io
        serialize(io, results)
    end

    results2 = do_PFI(
        flow_value, h_grid, a_grid, e_grid, trans_mat, β,
        δ=δ, α=α, η=η, A=A, γ=γ, θ=θ, ε=ε, z=z, r=r, tol=1f-6
    )

    open(joinpath(@__DIR__, "results", "PFI_results.jls"), "w") do io
        serialize(io, results2)
    end
end
