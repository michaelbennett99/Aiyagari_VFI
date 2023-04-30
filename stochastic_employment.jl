using BellmanSolver, NumericalMethods, Distributions, LinearAlgebra
using ProfileSVG

"""
The variables should be interpreted as follows:
    - c: consumption
    - l: leisure
    - e: employment
    - s: schooling
"""

function calculate_l(e::Int, s::Float64)
    return e + (1-e) * s
end

function calculate_s(
        h::Float64, hp::Float64, A::Float64, δ::Float64, α::Float64, η::Float64
    )
    return ((hp - (1-δ)*h)/(A*h^η))^(1/α)
end

function calculate_c(
        z::Float64, h::Float64, e::Int64,
        s::Float64, a::Float64, ap::Float64, r::Float64
    )
    return z + (h - z) * e * (1-s) + (1+r) * a - ap
end

function calculate_h(
        s::Float64, h::Float64, A::Float64, η::Float64, δ::Float64, α::Float64
    )
    return A * s^α * h^η + (1 - δ) * h
end

function u_c_l(c::Float64, l::Float64, θ::Float64, γ::Float64, ε::Float64)
    u_c(c) = θ != 1 ? c^(1-θ)/(1-θ) : log(c)
    u_l(l) = ε != -1 ? (ε/(1 + ε)) * l^((1+ε)/ε) : log(l)
    return u_c(c) - γ * u_l(l)
end

function flow_value(
        h::Float64, hp::Float64, a::Float64, ap::Float64, e::Int;
        z::Float64, A::Float64, δ::Float64, α::Float64,
        η::Float64, θ::Float64, γ::Float64, ε::Float64, r::Float64
        )
    s = calculate_s(h, hp, A, δ, α, η)
    l = calculate_l(e, s)
    c = calculate_c(z, h, e, s, a, ap, r)
    if c < 0 || s < 0 || s > 1
        return -Inf
    end
    return u_c_l(c, l, θ, γ, ε)
end

function calculate_emp_process(λ_f::Float64, λ_l::Float64)
    grid = [0, 1]
    trans = [1-λ_f λ_f; λ_l 1-λ_l]
    return (grid, trans)
end

function make_flow_value_mat(
        flow_value::Function,
        h_grid::Vector{Float64}, a_grid::Vector{Float64},
        e_grid::Vector{Int},
        hp_grid::Vector{Float64}, ap_grid::Vector{Float64};
        kwargs...
    )
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)
    hp_N = length(hp_grid)
    ap_N = length(ap_grid)

    flow_value_spec(h, hp, a, ap, e) = flow_value(h, hp, a, ap, e; kwargs...)

    flow_value_mat = Array{Float64}(undef, h_N, a_N, e_N, hp_N, ap_N)
    Threads.@threads for i_h ∈ 1:h_N
        for i_a ∈ 1:a_N, i_e ∈ 1:e_N, i_hp ∈ 1:hp_N, i_ap ∈ 1:ap_N
            flow_value_mat[i_h, i_a, i_e, i_hp, i_ap] = flow_value_spec(
                h_grid[i_h], hp_grid[i_hp], a_grid[i_a], ap_grid[i_ap],
                e_grid[i_e]
            )
        end
    end
    return flow_value_mat
end

function value_function(
        flow_value_mat::Array{Float64},
        V::Array{Float64},
        trans_mat::Matrix{Float64},
        i_h::Int, i_a::Int, i_e::Int,
        i_hp::Int, i_ap::Int, β::Float64
    )
    @views flow_val = flow_value_mat[i_h, i_a, i_e, i_hp, i_ap]
    @views ECont = V[i_hp, i_ap, :] ⋅ trans_mat[i_e, :]
    return flow_val + β * ECont
end

function do_VFI(
        flow_value::Function, h_grid::Vector{Float64}, a_grid::Vector{Float64},
        e_grid::Vector{Int}, trans_mat::Matrix{Float64}, β::Float64;
        tol::Float64=1e-6, max_iter::Int=1000, kwargs...
    )
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)

    # Set up grids

    ap_grid = copy(a_grid)
    hp_grid = copy(h_grid)

    V = zeros(h_N, a_N, e_N)
    hp_mat = Array{Float64}(undef, h_N, a_N, e_N)
    ap_mat = Array{Float64}(undef, h_N, a_N, e_N)

    println("Making flow value matrix ...")

    flow_value_mat = make_flow_value_mat(
        flow_value, h_grid, a_grid, e_grid, hp_grid, ap_grid; kwargs...
    )

    value_fn_spec(V, i_h, i_a, i_e, i_hp, i_ap) = value_function(
        flow_value_mat, V, trans_mat, i_h, i_a, i_e, i_hp, i_ap, β
    )

    println("Starting iteration ...")

    diff = 1
    iter = 0
    while diff > tol && iter < max_iter
        val_mat = Array{Float64}(undef, h_N, a_N, e_N)
        Threads.@threads for i_h ∈ 1:h_N
            for i_a ∈ 1:a_N, i_e ∈ 1:e_N
                val::Float64 = -Inf
                val_hp::Float64 = NaN
                val_ap::Float64 = NaN
                for i_hp ∈ 1:h_N, i_ap ∈ 1:a_N
                    @views candidate_val = value_fn_spec(
                        V, i_h, i_a, i_e, i_hp, i_ap
                    )
                    if candidate_val > val
                        val = candidate_val
                        val_hp = hp_grid[i_hp]
                        val_ap = ap_grid[i_ap]
                    end
                end
                val_mat[i_h, i_a, i_e] = val
                hp_mat[i_h, i_a, i_e] = val_hp
                ap_mat[i_h, i_a, i_e] = val_ap
            end
        end
        diff = maximum(abs.(V - val_mat))
        V = val_mat
        if iter % 25 == 0
            println("Iteration $iter. Diff = $diff.")
        end
        iter += 1
    end
    return h_grid, a_grid, e_grid, V, hp_mat, ap_mat, iter
end
