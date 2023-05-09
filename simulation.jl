#!/usr/bin/env julia

using Interpolations, Distributions, PlotlyJS

struct TimePath
    T::Int
    a::Vector{Float32}
    h::Vector{Float32}
    e::Vector{Float32}
end

struct Population
    N::Int
    time_paths::Vector{TimePath}
end

function simulate_time_path(
        a_grid, h_grid, hp, ap, trans_mat; T=100, a_0 = 0f0, h_0 = 1f0
    )
    hp_itp = [
        interpolate(hp[:, :, x], BSpline(Cubic(Line(OnGrid()))))
        for x ∈ axes(hp, 3)
    ]
    ap_itp = [
        interpolate(ap[:, :, x], BSpline(Cubic(Line(OnGrid()))))
        for x ∈ axes(ap, 3)
    ]
    h_range = range(minimum(h_grid), maximum(h_grid), length(h_grid))
    a_range = range(minimum(a_grid), maximum(a_grid), length(a_grid))
    hp_fn = [
        extrapolate(
            Interpolations.scale(x, h_range, a_range), Line()
        ) for x ∈ hp_itp
    ]
    ap_fn = [
        extrapolate(
            Interpolations.scale(x, h_range, a_range), Line()
        ) for x ∈ ap_itp
    ]
    a = Vector{Float32}(undef, T)
    h = Vector{Float32}(undef, T)
    e = Vector{Int}(undef, T)
    p_0 = 1 - (trans_mat[2, 1] / (trans_mat[2, 1] + trans_mat[1, 2]))
    e_0 = Int(rand(Bernoulli(p_0)))
    a[1] = a_0
    h[1] = h_0
    e[1] = e_0
    for t in 2:T
        e_ix = e[t-1] + 1
        a[t] = ap_fn[e_ix](h[t-1], a[t-1])
        h[t] = hp_fn[e_ix](h[t-1], a[t-1])
        e[t] = Int(rand(Bernoulli(trans_mat[e_ix, 2])))
    end
    return TimePath(T, a, h, e)
end

function simulate_population(
        a_grid, h_grid, hp, ap, trans_mat; 
        N=1000, T=100, a_0 = 0f0, h_0 = 1f0
    )
    time_paths = Vector{TimePath}(undef, N)
    Threads.@threads for i ∈ 1:N
        time_paths[i] = simulate_time_path(
            a_grid, h_grid, hp, ap, trans_mat; T=T, a_0=a_0, h_0=h_0
        )
    end
    return Population(N, time_paths)
end

function make_density_plot(population::Population, t::Int)
    x = [z.a[t] for z ∈ population.time_paths]
    y = [z.h[t] for z ∈ population.time_paths]

    trace1 = histogram2dcontour(
            x = x,
            y = y,
            colorscale = "Blues",
            reversescale = true,
            xaxis = "x",
            yaxis = "y"
        )
    trace2 = scatter(
            x = x,
            y = y,
            xaxis = "x",
            yaxis = "y",
            mode = "markers",
            marker = attr(
                color = "rgba(0,0,0,0.1)",
                size = 1
            )
        )
    trace3 = histogram(
            y = y,
            xaxis = "x2",
            marker = attr(
                color = "rgba(0,0,0,1)"
            )
        )
    trace4 = histogram(
            x = x,
            yaxis = "y2",
            marker = attr(
                color = "rgba(0,0,0,1)"
            )
        )

    layout = Layout(
        autosize = false,
        xaxis = attr(
            zeroline = false,
            domain = [0,0.85],
            showgrid = false
        ),
        yaxis = attr(
            zeroline = false,
            domain = [0,0.85],
            showgrid = false
        ),
        xaxis2 = attr(
            zeroline = false,
            domain = [0.85,1],
            showgrid = false
        ),
        yaxis2 = attr(
            zeroline = false,
            domain = [0.85,1],
            showgrid = false
        ),
        height = 600,
        width = 600,
        bargap = 0,
        hovermode = "closest",
        showlegend = false,
        xaxis_title = "Assets",
        yaxis_title = "Human Capital"
    )

    p = plot([trace1, trace2, trace3, trace4], layout)

    svgfile = joinpath(@__DIR__, "plots", "density_plot/$t.pdf")
    touch(svgfile)
    open(svgfile, "w") do io
        savefig(io, p.plot, format="pdf")
    end
    return p
end

function make_density_plots(population::Population)
    plots = Vector{Any}(undef, population.time_paths[1].T)
    for t ∈ 1:population.time_paths[1].T
        plots[t] = make_density_plot(population, t)
    end
end
