#!/usr/bin/env julia
using Serialization

include("stochastic_employment.jl")

# Set parameters
const β = 0.9f0

const δ = 0.1f0
const α  = 1f0
const η = 0f0
const A = 1.5f0

const γ = 1f0
const θ = 1.5f0
const ε = 0.5f0

const z = 1f0
const r = 0.1f0

const λ_f = 0.95f0
const λ_l = 0.05f0

# Set up grid lengths
const h_N = 200
const a_N = 200
const e_N = 2

# Define parameters
const a_lbar = -z / r * 0.99f0
const h_bar = (A /  δ)^(1 / (1-η))

# Set up grids
const h_grid = collect(Float32, range(z, h_bar, h_N))
const a_grid = collect(Float32, range(a_lbar, -a_lbar, a_N))
const e_grid, trans_mat = calculate_emp_process(λ_f, λ_l)

if !isinteractive()
    results = do_VFI(
        flow_value, h_grid, a_grid, e_grid, trans_mat, β,
        δ=δ, α=α, η=η, A=A, γ=γ, θ=θ, ε=ε, z=z, r=r, tol=1f-6
    )

    if !isdir("results")
        mkdir("results")
    end
    touch("results/VFI_results.jls")
    open(joinpath(@__DIR__, "results", "VFI_results.jls"), "w") do io
        serialize(io, results)
    end
end
