using PlotlyJS, PlotlyBase

const m = 0

function make_3d_plot(
        h_grid, a_grid, outcome_mats, names, title, zax_title, camera, out;
        include_zero=false
    )
    layout = Layout(
        showlegend=true,
        autosize=false,
        width=700, height=600, margin=attr(l=0, r=0, b=0, t=0),
        scene=attr(
            xaxis_title="Human Capital",
            yaxis_title="Wealth",
            zaxis_title=zax_title,
            legend_title="Employment",
            aspectmode="cube",
            camera=camera
        )
    )

    colourscale = ["Viridis", "RdBu"]

    outcome_plots = [
            surface(
                x=h_grid, y=a_grid, z=outcome_mat, name=name,
                colorscale=cscale, showscale=false, showlegend=true
            ) for (outcome_mat, name, cscale)
            ∈ zip(outcome_mats, names, colourscale)
        ]

    if include_zero
        push!(
            outcome_plots,
            surface(
                x=h_grid, y=a_grid, z=zeros(size(outcome_mats[1])),
                name="Zero", colorscale="Greys", showscale=false, opacity=0.5
            )
        )    
    end

    p = plot(
        outcome_plots, layout
    )

    filename = joinpath(@__DIR__, "plots", "$out.html")
    svgfile = joinpath(@__DIR__, "plots", "$out.pdf")
    touch(filename)
    open(filename, "w") do io
        PlotlyBase.to_html(io, p.plot)
    end
    touch(svgfile)
    open(svgfile, "w") do io
        savefig(io, p.plot, format="pdf")
    end
    return p
end
