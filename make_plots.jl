using PlotlyJS

function make_3d_plot(
        h_grid, a_grid, outcome_mats, names, title, zax_title, out;
        include_zero=false
    )
    layout = Layout(
        title=title,
        scene=attr(
            xaxis_title="Human Capital",
            yaxis_title="Wealth",
            zaxis_title=zax_title
        )
    )

    colourscale = ["Viridis", "RdBu"]

    outcome_plots = [
            surface(
                x=h_grid, y=a_grid, z=outcome_mat, name=name,
                colorscale=cscale, showscale=false
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

    touch("$out.html")
    open("$out.html", "w") do io
        PlotlyBase.to_html(io, p.plot)
    end
    return p
end
