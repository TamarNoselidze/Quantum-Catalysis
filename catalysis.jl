using JLD2

include("helper.jl")
@load "simplex_dataset_big.jld2" dataset



# The catalyst
catalyst = [0.6, 0.4]

# # The incomparable states
# x = [0.4, 0.4, 0.1, 0.1]
# y = [0.5, 0.25, 0.25, 0.0]

# # Without catalyst
# println(is_majorized(x, y))

# println(is_catalysis_possible(x, y, c))


function run_catalysis_simulation(dataset, catalyst)
    plain_locc_impossible_count = 0
    catalysis_possible_count = 0



    n_samples = size(dataset, 2)

    for i in 1:n_samples
        for j in (i + 1):n_samples  
            x = @views dataset[:, i]
            y = @views dataset[:, j]

            if !is_locc_convertible(x, y) 
                plain_locc_impossible_count += 1
                        
                if is_catalysis_possible(x, y, catalyst)
                    catalysis_possible_count += 1
                end
                        
            end 
        end 
    end 

    println("Plain LOCC impossible: ", plain_locc_impossible_count)
    println("Catalysis helped: ", catalysis_possible_count)
end


# run_catalysis_simulation(dataset, catalyst)



function plot_distance_histogram(dataset; num_bins=10)
    d, n_samples = size(dataset)

    # maximally entangled state / center
    center = fill(1.0 / d, d)

    # squared distances for all samples
    sq_distances = Float64[]
    for i in 1:n_samples
        state = @views dataset[:, i]
        # sum(abs2, ...) for the squared Euclidean distance
        sq_dist = sum(abs2, state .- center) 
        push!(sq_distances, sq_dist)
    end


    empirical_mean = sum(sq_distances) / n_samples
    theoretical_mean = (d - 1) / (d * (d + 1))

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


plot_distance_histogram(dataset)