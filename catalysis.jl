using JLD2

include("helper_functions.jl")
@load "dataset_20k_d7.jld2" dataset



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


    n_samples = size(dataset, 2) # returns size of dimension 2 of the dataset matrix, i.e., number of columns (sampled states)

    for i in 1:n_samples
        for j in (i + 1):n_samples  # Nested loop over all unique unordered pairs, i < j only
            
            # @views avoids creating an array copy of column i of dataset; it creates a lightweight reference into the matrix instead
            x = @views dataset[:, i]    # 'view all row indices from column i of dataset'
            y = @views dataset[:, j]

            if !is_locc_convertible(x, y) 
                plain_locc_impossible_count += 1

                # testing for catalysis in both directions        
                if is_catalysis_possible(x, y, catalyst)
                    catalysis_possible_count += 1
                end
                        
            end 
        end 
    end 

    println("Pairs with impossible plain LOCC conversion: ", plain_locc_impossible_count)
    println("Pairs with conversion enabled by catalyst: ", catalysis_possible_count)
end


# run_catalysis_simulation(dataset, catalyst)



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

    # #Sanity check: relative difference between empirical and theoretical mean
    # rel_diff = abs(empirical_mean - theoretical_mean) / theoretical_mean
    # if rel_diff < tol
    #     println("Sampling looks uniform on the simplex.")
    # else
    #     println("Sanity check for sampling uniformity failed.")
    # end


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