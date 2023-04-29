using BellmanSolver, NumericalMethods, Distributions, LinearAlgebra

"""
The variables should be interpreted as follows:
    - c: consumption
    - l: leisure
    - e: employment
    - s: schooling
"""

function calculate_l(e::Integer, s::Real)
    return e + (1-e) * s
end

function calculate_s(h::Real, hp::Real, A::Real, δ::Real, α::Real, η::Real)
    return ((hp - (1-δ)*h)/(A*h^η))^(1/α)
end

function calculate_c(
        z::Real, h::Real, e::Real, s::Real, a::Real, ap::Real, r::Real
    )
    return z + (h - z) * e * (1-s) + (1+r) * a - ap
end

function calculate_h(s::Real, h::Real, A::Real, η::Real, δ::Real, α::Real)
    return A * s^α * h^η + (1 - δ) * h
end

function u_c_l(c::Real, l::Real, θ::Real, γ::Real, ε::Real)
    u_c(c) = θ != 1 ? c^(1-θ)/(1-θ) : log(c)
    u_l(l) = ε != -1 ? (ε/(1 + ε)) * l^((1+ε)/ε) : log(l)
    return u_c(c) - γ * u_l(l)
end

function flow_value(
        h::Real, hp::Real, a::Real, ap::Real, e::Integer;
        z::Real, A::Real, δ::Real, α::Real,
        η::Real, θ::Real, γ::Real, ε::Real, r::Real
        )
    s = calculate_s(h, hp, A, δ, α, η)
    l = calculate_l(e, s)
    c = calculate_c(z, h, e, s, a, ap, r)
    if c < 0 || s < 0 || s > 1
        return -Inf
    end
    return (1/(1-θ)) * c^(1 - θ) + γ * (ε/(1+ε)) * l^((1+ε)/ε)
end

function calculate_emp_process(λ_f::Real, λ_l::Real)
    grid = [0, 1]
    trans = [1-λ_f λ_f; λ_l 1-λ_l]
    return (grid, trans)
end

function make_flow_value_mat(
        flow_value::Function,
        h_grid::Vector{<:Real}, a_grid::Vector{<:Real},
        e_grid::Vector{<:Integer},
        hp_grid::Vector{<:Real}, ap_grid::Vector{<:Real};
        kwargs...
    )
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)
    hp_N = length(hp_grid)
    ap_N = length(ap_grid)

    flow_value_mat = Array{Float64}(undef, h_N, a_N, e_N, hp_N, ap_N)
    for (i_h, h) ∈ enumerate(h_grid),
        (i_a, a) ∈ enumerate(a_grid),
        (i_e, e) ∈ enumerate(e_grid),
        (i_hp, hp) ∈ enumerate(hp_grid),
        (i_ap, ap) ∈ enumerate(ap_grid)
        flow_value_mat[i_h, i_a, i_e, i_hp, i_ap] = flow_value(
            h, hp, a, ap, e; kwargs...
        )
    end
    return flow_value_mat
end

function value_function(
        flow_value_mat::AbstractArray{<:Real},
        V::AbstractArray{<:Real},
        trans_mat::AbstractArray{<:Real},
        i_h::Integer, i_a::Integer, i_e::Integer,
        i_hp::Integer, i_ap::Integer, β::Real
    )
    @views flow_val = flow_value_mat[i_h, i_a, i_e, i_hp, i_ap]
    @views ECont = V[i_hp, i_ap, :] ⋅ trans_mat[i_e, :]
    return flow_val + β * ECont
end

function do_VFI(
        flow_value::Function, h_grid::Vector{<:Real}, a_grid::Vector{<:Real},
        e_grid::Vector{<:Integer}, trans_mat::Matrix{<:Real}, β::Real;
        tol::Float64=1e-6, max_iter::Int=1000, kwargs...
    )
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)

    # Set up grids

    ap_grid = copy(a_grid)
    hp_grid = copy(h_grid)

    V = Array{Float64}(undef, h_N, a_N, e_N)
    hp_mat = Array{Float64}(undef, h_N, a_N, e_N)
    ap_mat = Array{Float64}(undef, h_N, a_N, e_N)

    println("Making flow value matrix ...")

    flow_value_mat = make_flow_value_mat(
        flow_value, h_grid, a_grid, e_grid, hp_grid, ap_grid; kwargs...
    )

    println("Starting iteration ...")

    diff = 1
    iter = 0
    while diff > tol && iter < max_iter
        val_mat = Array{Float64}(undef, h_N, a_N, e_N)
        for (i_h, i_a, i_e) ∈ Iterators.product(1:h_N, 1:a_N, 1:e_N)
            val = -Inf
            val_hp = NaN
            val_ap = NaN
            for (i_hp, i_ap) ∈ Iterators.product(1:h_N, 1:a_N)
                candidate_val = value_function(
                    flow_value_mat, V, trans_mat, i_h, i_a, i_e, i_hp, i_ap, β
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
        diff = maximum(abs.(V - val_mat))
        V = val_mat
        iter += 1
        if iter % 10 == 0
            println("Iteration $iter. Diff = $diff.")
        end
    end
    return h_grid, a_grid, e_grid, V, hp_mat, ap_mat, iter
end
