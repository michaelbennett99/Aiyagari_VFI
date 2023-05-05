using BellmanSolver, Distributions, LinearAlgebra
using Interpolations, Optim, ForwardDiff

include("NumericalMethods.jl")

set_zero_subnormals(true)

"""
The variables should be interpreted as follows:
    - c: consumption
    - l: leisure
    - e: employment
    - s: schooling
"""

function calculate_l(e::Int, s::Float32)::Float32
    return e + (1-e) * s
end

function calculate_s(
        h::Float32, hp::Float32, A::Float32, δ::Float32, α::Float32, η::Float32
    )::Float32
    return ((hp - (1-δ)*h)/(A*h^η))^(1/α)
end

function calculate_c(
        z::Float32, h::Float32, e::Int64,
        s::Float32, a::Float32, ap::Float32, r::Float32
    )
    return z + (h - z) * e * (1-s) + (1+r) * a - ap
end

function calculate_h(
        s::Float32, h::Float32, A::Float32, η::Float32, δ::Float32, α::Float32
    )::Float32
    return A * s^α * h^η + (1 - δ) * h
end

function u_c_l(
        c::Float32, l::Float32, θ::Float32, γ::Float32, ε::Float32
    )::Float32
    u_c(c) = θ != 1 ? c^(1-θ)/(1-θ) : log(c)
    u_l(l) = ε != -1 ? (ε/(1 + ε)) * l^((1+ε)/ε) : log(l)
    return u_c(c) - γ * u_l(l)
end

function flow_value(
        h::Float32, hp::Float32, a::Float32, ap::Float32, e::Int;
        z::Float32, A::Float32, δ::Float32, α::Float32,
        η::Float32, θ::Float32, γ::Float32, ε::Float32, r::Float32
    )::Float32
    s = calculate_s(h, hp, A, δ, α, η)
    l = calculate_l(e, s)
    c = calculate_c(z, h, e, s, a, ap, r)
    if c < 0 || s < 0 || s > 1
        return -Inf32
    end
    return u_c_l(c, l, θ, γ, ε)
end

function calculate_emp_process(λ_f::Float32, λ_l::Float32)
    grid = [0, 1]
    trans = [1-λ_f λ_f; λ_l 1-λ_l]
    return (grid, trans)
end

function make_flow_value_mat(
        flow_value::Function,
        h_grid::Vector{Float32}, a_grid::Vector{Float32},
        e_grid::Vector{Int},
        hp_grid::Vector{Float32}, ap_grid::Vector{Float32};
        kwargs...
    )::Array{Float32, 5}
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)
    hp_N = length(hp_grid)
    ap_N = length(ap_grid)

    flow_value_spec(h, hp, a, ap, e) = flow_value(h, hp, a, ap, e; kwargs...)

    flow_value_mat = Array{Float32}(undef, h_N, a_N, e_N, hp_N, ap_N)
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
        flow_value_mat::Array{Float32},
        V::Array{Float32},
        trans_mat::Matrix{Float32},
        i_h::Int, i_a::Int, i_e::Int,
        i_hp::Int, i_ap::Int, β::Float32
    )::Float32
    @views @inbounds flow_val = flow_value_mat[i_h, i_a, i_e, i_hp, i_ap]
    @views @inbounds @fastmath ECont = V[i_hp, i_ap, :] ⋅ trans_mat[i_e, :]
    return flow_val + β * ECont
end

function max_V_ix(
        value_fn::Function, V::Array{Float32}, i_h::Int, i_a::Int, i_e::Int
    )::Float32
    val::Float32 = -Inf32
    for i_hp ∈ 1:h_N, i_ap ∈ 1:a_N
        @views candidate_val = value_fn(
            V, i_h, i_a, i_e, i_hp, i_ap
        )
        if candidate_val > val
            val = candidate_val
        end
    end
    return val
end

function max_V_ix_final(
        value_fn::Function, V::Array{Float32}, i_h::Int, i_a::Int, i_e::Int
    )::Tuple{Float32, Int, Int}
    val::Float32 = -Inf32
    val_i_hp::Int = 0
    val_i_ap::Int = 0
    for i_hp ∈ 1:h_N, i_ap ∈ 1:a_N
        @views candidate_val = value_fn(
            V, i_h, i_a, i_e, i_hp, i_ap
        )
        if candidate_val > val
            val = candidate_val
            val_i_hp = i_hp
            val_i_ap = i_ap
        end
    end
    return val, val_i_hp, val_i_ap
end

function update_val_mat!(
        val_mat::Array{Float32}, V::Array{Float32}, value_fn::Function,
        h_iterator::UnitRange{Int64}, a_iterator::UnitRange{Int64},
        e_iterator::UnitRange{Int64}
    )
    Threads.@threads for i_h ∈ h_iterator
        for i_a ∈ a_iterator, i_e ∈ e_iterator
            val_mat[i_h, i_a, i_e] = max_V_ix(
                value_fn, V, i_h, i_a, i_e
            )
        end
    end
end

function get_val_arg_mats(
        V::Array{Float32}, value_fn::Function, h_iterator::UnitRange{Int64},
        a_iterator::UnitRange{Int64}, e_iterator::UnitRange{Int64}
    )::Tuple{Array{Float32}, Array{Int}, Array{Int}}
    val_mat = Array{Float32}(undef, h_N, a_N, e_N)
    i_hp_mat = Array{Int}(undef, h_N, a_N, e_N)
    i_ap_mat = Array{Int}(undef, h_N, a_N, e_N)
    Threads.@threads for i_h ∈ h_iterator
        for i_a ∈ a_iterator, i_e ∈ e_iterator
            val, val_i_hp, val_i_ap = max_V_ix_final(
                value_fn, V, i_h, i_a, i_e
            )
            val_mat[i_h, i_a, i_e] = val
            i_hp_mat[i_h, i_a, i_e] = val_i_hp
            i_ap_mat[i_h, i_a, i_e] = val_i_ap
        end
    end
    return val_mat, i_hp_mat, i_ap_mat
end

function do_VFI(
        flow_value::Function, h_grid::Vector{Float32}, a_grid::Vector{Float32},
        e_grid::Vector{Int}, trans_mat::Matrix{Float32}, β::Float32;
        tol::Float32=1f-3, max_iter::Int=1000, kwargs...
    )::Tuple{
        Vector{Float32}, Vector{Float32}, Vector{Int}, Array{Float32},
        Array{Float32}, Array{Float32}, Int
    }
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)

    # Set up grids

    println("Making flow value matrix ...")

    flow_value_mat = make_flow_value_mat(
        flow_value, h_grid, a_grid, e_grid, h_grid, a_grid; kwargs...
    )

    V = maximum(flow_value_mat, dims=(4, 5))

    value_fn_spec(V, i_h, i_a, i_e, i_hp, i_ap) = value_function(
        flow_value_mat, V, trans_mat, i_h, i_a, i_e, i_hp, i_ap, β
    )

    println("Starting iteration ...")

    diff = 1
    iter = 0
    val_mat = Array{Float32}(undef, h_N, a_N, e_N)
    while diff > tol && iter < max_iter
        update_val_mat!(val_mat, V, value_fn_spec, 1:h_N, 1:a_N, 1:e_N)
        @fastmath diff = maximum(abs.(V - val_mat))
        V = copy(val_mat)
        if iter % 25 == 0
            println("Iteration $iter. Diff = $diff.")
        end
        iter += 1
    end
    V, i_hp_mat, i_ap_mat = get_val_arg_mats(
        V, value_fn_spec, 1:h_N, 1:a_N, 1:e_N
    )
    hp_mat = map(x -> h_grid[x], i_hp_mat)
    ap_mat = map(x -> a_grid[x], i_ap_mat)
    return h_grid, a_grid, e_grid, V, hp_mat, ap_mat, iter
end

function get_valid_subarray(array::AbstractMatrix)
    col_sums = [sum(.!isinf.(array[:, j])) for j ∈ axes(array, 2)]
    valid_cols = Bool.([y == maximum(col_sums) for y ∈ col_sums])
    valid_rows = Bool.(
        [.!all(isinf.(array[j, valid_cols])) for j ∈ axes(array, 1)]
    )
    return valid_rows, valid_cols
end

function do_VFI_I(
        flow_value::Function, h_grid::Vector{Float32}, a_grid::Vector{Float32},
        e_grid::Vector{Int}, trans_mat::Matrix{Float32}, β::Float32;
        tol::Float32=1f-3, max_iter::Int=1000, kwargs...
    )
    h_N = length(h_grid)
    a_N = length(a_grid)
    e_N = length(e_grid)

    i_h_grid = 1:h_N
    i_a_grid = 1:a_N
    i_e_grid = 1:e_N

    # Set up grids

    println("Making flow value matrix ...")

    flow_value_mat = make_flow_value_mat(
        flow_value, h_grid, a_grid, e_grid, h_grid, a_grid; kwargs...
    )

    # Set up guess for value function

    V = maximum(flow_value_mat, dims=(4, 5))
    hp_mat = Array{Float32}(undef, h_N, a_N, e_N)
    ap_mat = Array{Float32}(undef, h_N, a_N, e_N)

    value_fn_spec(V, i_h, i_a, i_e, i_hp, i_ap) = value_function(
        flow_value_mat, V, trans_mat, i_h, i_a, i_e, i_hp, i_ap, β
    )

    println("Starting iteration ...")

    odiff = 0
    diff = 1
    iter = 0
    while diff > tol && iter < max_iter && odiff != diff
        val_mat = Array{Float32}(undef, h_N, a_N, e_N)
        Threads.@threads for i_h ∈ i_h_grid
            for i_a ∈ i_a_grid, i_e ∈ i_e_grid
                V_i = Matrix{Float32}(undef, h_N, a_N)
                for i_hp ∈ i_h_grid, i_ap ∈ i_a_grid
                    @views @inbounds V_i[i_hp, i_ap] = value_fn_spec(
                        V, i_h, i_a, i_e, i_hp, i_ap
                    )
                end
                valid_rows, valid_cols = get_valid_subarray(V_i)
                if sum(valid_cols) == 1 && sum(valid_rows) == 1
                    val_mat[i_h, i_a, i_e] = V_i[valid_rows, valid_cols][1, 1]
                    hp_mat[i_h, i_a, i_e] = vec(h_grid[valid_rows])[1]
                    ap_mat[i_h, i_a, i_e] = vec(a_grid[valid_cols])[1]
                elseif sum(valid_cols) == 1
                    f_V_i = vec(V_i[valid_rows, valid_cols])
                    f_i = vec(h_grid[valid_rows])
                    itp = interpolate(f_V_i, BSpline(Cubic(Line(OnGrid()))))
                    V_i_fn = extrapolate(
                        Interpolations.scale(
                            itp,
                            range(minimum(f_i), maximum(f_i), length(f_i))),
                        Line()
                    )
                    obj = x -> -V_i_fn(x)
                    max = optimize(obj, f_i[1], f_i[end])
                    val_mat[i_h, i_a, i_e] = -Optim.minimum(max)
                    hp_mat[i_h, i_a, i_e] = Optim.minimizer(max)
                    ap_mat[i_h, i_a, i_e] = vec(a_grid[valid_cols])[1]
                elseif sum(valid_rows) == 1
                    f_V_i = vec(V_i[valid_rows, valid_cols])
                    f_i = vec(a_grid[valid_cols])
                    itp = interpolate(f_V_i, BSpline(Cubic(Line(OnGrid()))))
                    V_i_fn = extrapolate(
                        Interpolations.scale(
                            itp, range(minimum(f_i), maximum(f_i), length(f_i))),
                        Line()
                    )
                    obj = x -> -V_i_fn(x)
                    max = optimize(obj, f_i[1], f_i[end])
                    val_mat[i_h, i_a, i_e] = -Optim.minimum(max)
                    hp_mat[i_h, i_a, i_e] = vec(h_grid[valid_rows])[1]
                    ap_mat[i_h, i_a, i_e] = Optim.minimizer(max)
                else
                    f_V_i = V_i[valid_rows, valid_cols]
                    f_i_h = h_grid[valid_rows]
                    f_i_a = a_grid[valid_cols]
                    itp = interpolate(f_V_i, BSpline(Cubic(Line(OnGrid()))))
                    V_i_fn = extrapolate(Interpolations.scale(
                        itp,
                        range(minimum(f_i_h), maximum(f_i_h), length(f_i_h)),
                        range(minimum(f_i_a), maximum(f_i_a), length(f_i_a))
                    ), Interpolations.Flat())
                    obj = x -> -V_i_fn(x[1], x[2])
                    max = optimize(obj, [f_i_h[ceil(Int, sum(valid_rows)/2)], f_i_a[ceil(Int, sum(valid_cols)/2)]])
                    val_mat[i_h, i_a, i_e] = -Optim.minimum(max)
                    hp_mat[i_h, i_a, i_e] = Optim.minimizer(max)[1]
                    ap_mat[i_h, i_a, i_e] = Optim.minimizer(max)[2]
                end
            end
        end
        odiff = copy(diff)
        diff = maximum(abs.(V - val_mat))
        V = val_mat
        if iter % 25 == 0
            println("Iteration $iter. Diff = $diff.")
        end
        iter += 1
    end
    return h_grid, a_grid, e_grid, V, hp_mat, ap_mat, iter
end
