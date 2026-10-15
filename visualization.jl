function plot_2D_catalysis_results(results_dict, dim, total_pairs)
    # extract p value from the [p, 1-p] catalyst vector
    data = [(cat[1], count) for (cat, count) in results_dict]
    
    sort!(data, by = x -> x[1])
    
    # split into x and y arrays for plotting
    catalysts_x = [d[1] for d in data]
    success_counts = [d[2] for d in data]
    
    p = plot(catalysts_x, success_counts, 
             xlabel="Catalyst Parameter (p)", 
             ylabel="Enabled Conversions", 
             title="Catalysis Efficiency vs. Parameter p ($total_pairs total pairs)",
             linewidth=2, marker=:circle, label="d = $dim",
             legend=:topright)
             
    display(p)
    savefig(p, "./plots/catalysis_curve_d$dim.png")
    # println("Plot saved as catalysis_curve_d$dim.png")
end



#Purpose: plot the 3D catalysis results as a 2D scatter plot (since 3D catalyst has only 2 free parameters)
#with color representing success count
function plot_3D_catalysis_results(results_dict, dim, total_pairs)
    # Extract p1, p2, and the success count
    p1_vals = [cat[1] for (cat, count) in results_dict]
    p2_vals = [cat[2] for (cat, count) in results_dict]
    success_counts = [count for (cat, count) in results_dict]
    
    # Create a scatter plot where the color represents the success count
    p = scatter(p1_vals, p2_vals, zcolor=success_counts,
                xlabel="Catalyst Parameter p1", 
                ylabel="Catalyst Parameter p2",
                title="3D Catalysis Efficiency ($total_pairs pairs)",
                linewidth=2, marker=:circle, label="d = $dim",
                markersize=5, markerstrokewidth=0,
                colorbar_title="Enabled Conversions",
                legend=:topright, aspect_ratio=:equal)
             
    display(p)
    savefig(p, "./plots/catalysis_heatmap_3D_d$dim.png")
    # println("Plot saved as catalysis_heatmap_3D_d$dim.png")
end


function plot_4D_catalysis_results(results_dict, dim, total_pairs)
    p1_vals = [cat[1] for (cat, count) in results_dict]
    p2_vals = [cat[2] for (cat, count) in results_dict]
    p3_vals = [cat[3] for (cat, count) in results_dict]
    success_counts = [count for (cat, count) in results_dict]
    
    p = scatter3d(p1_vals, p2_vals, p3_vals, zcolor=success_counts,
                  xlabel="Parameter p1", 
                  ylabel="Parameter p2",
                  zlabel="Parameter p3",
                  title="4D Catalysis Efficiency ($total_pairs pairs)",
                  marker=:circle, markersize=3, markerstrokewidth=0,
                  colorbar_title="Enabled Conversions",
                  legend=false,
                  camera=(45, 30))
             
    # Save as HTML for full Plotly interactivity
    savefig(p, "./plots/catalysis_scatter3D_d$(dim)_4D.html")
    # println("Interactive plot saved as catalysis_scatter3D_d$(dim)_4D.html")
end




function plot_distance_histogram(dataset; num_bins=10, tol=0.05)
    d, n_samples = size(dataset)     # d takes first dimensions (rows), n_samples takes second (columns)

    println("Dimension of the dataset: ", d)
    println("Number of states in the dataset: ", n_samples)

    # maximally entangled state / center
    center = fill(1.0 / d, d)

    # squared distances for all samples
    sq_distances = Float64[]
    for i in 1:n_samples
        state = @views dataset[:, i]
        # sum(abs2, ...) for the squared Euclidean distances from each sampled point to the simplex's center
        sq_dist = sum(abs2, state .- center)
        push!(sq_distances, sq_dist)  # Add sq_dist to the end of the sq_distances array.
    end

    empirical_mean = sum(sq_distances) / n_samples
    theoretical_mean = (d - 1) / (d * (d + 1))   # Standard variance formula for a symmetric Dirichlet(1,...,1) distribution

    
    println("Theoretical Mean Sq Distance: ", round(theoretical_mean, digits=4))
    println("Empirical Mean Sq Distance: ", round(empirical_mean, digits=4))

    min_dist = minimum(sq_distances)
    max_dist = maximum(sq_distances)
    bin_width = (max_dist - min_dist) / num_bins
    bins = zeros(Int, num_bins)
    for dist in sq_distances
        bin_idx = floor(Int, (dist - min_dist) / bin_width) + 1

        if bin_idx > num_bins
            bin_idx = num_bins
        end
        bins[bin_idx] += 1
    end

    println("SQUARED DISTANCE HISTOGRAM")
    for i in 1:num_bins
        bin_start = round(min_dist + (i - 1) * bin_width, digits=3)
        bin_end = round(min_dist + i * bin_width, digits=3)
        count = bins[i]
        bar_length = round(Int, (count / maximum(bins)) * 40)
        bar = repeat("=", bar_length)

        println("[$bin_start to $bin_end]: $count \t | $bar")
    end
end



function plot_catalysis_results(results_dict, state_dim, total_count, catalyst_dim)
    if catalyst_dim == 2
        plot_2D_catalysis_results(results_dict, state_dim, total_count)
    elseif catalyst_dim == 3
        plot_3D_catalysis_results(results_dict, state_dim, total_count)
    elseif catalyst_dim == 4
        plot_4D_catalysis_results(results_dict, state_dim, total_count)
    end
end