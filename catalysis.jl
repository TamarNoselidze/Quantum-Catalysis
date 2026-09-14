using JLD2
using Plots

include("helper_functions.jl")


function run_catalysis_simulation(left_side, right_side, catalyst)
    plain_locc_impossible_count = 0
    catalysis_possible_count = 0


    n_samples = size(left_side, 2) # returns size of dimension 2 of the dataset matrix, i.e., number of columns (sampled states)

    for i in 1:n_samples
        # @views avoids creating an array copy of column i of dataset; it creates a lightweight reference into the matrix instead
        x = @views left_side[:, i]    # 'view all row indices from column i of dataset'
        y = @views right_side[:, i]

        if !is_locc_convertible(x, y) 
            plain_locc_impossible_count += 1

            # testing for catalysis in both directions        
            if is_catalysis_possible(x, y, catalyst)
                catalysis_possible_count += 1
            end
                    
        end 
    end 

    # println("  Pairs with impossible plain LOCC conversion: ", plain_locc_impossible_count)
    # println("  Pairs with conversion enabled by catalyst: ", catalysis_possible_count)
    return plain_locc_impossible_count, catalysis_possible_count
end


function sample_catalysts(step=0.01, dimension=2)
    catalysts = Vector{Float64}[]
    for p in 0.0:step:1.0
        catalyst = [round(p, digits=4), round(1 - p, digits=4)]
        push!(catalysts, catalyst)
    end 
    return catalysts
end


function plot_catalysis_results(results_dict)
    # Extract tuples of (p_value, count) from the dictionary
    # cat[1] takes the `p` value from your [p, 1-p] catalyst vector
    data = [(cat[1], count) for (cat, count) in results_dict]
    
    # Sort the data by the p_value (x-axis) so the plot line connects sequentially!
    sort!(data, by = x -> x[1])
    
    # Split back into x and y arrays for plotting
    catalysts_x = [d[1] for d in data]
    success_counts = [d[2] for d in data]
    
    p = plot(catalysts_x, success_counts, 
             xlabel="Catalyst Parameter (p)", 
             ylabel="Enabled Conversions", 
             title="Catalysis Efficiency vs. Parameter p",
             linewidth=2, marker=:circle, label="d = 5",
             legend=:topright)
             
    display(p)
    savefig(p, "catalysis_curve_d5.png")
    println("Plot saved as catalysis_curve_d5.png")
end


if abspath(PROGRAM_FILE) == @__FILE__   
    d = 5
    @load "dataset_20k_d$d.jld2" dataset


    dataset_l = dataset[:, 1:10000] 
    dataset_r = dataset[:, 10001:20000]
    catalysts = sample_catalysts()


    results_dict = Dict{Vector{Float64}, Int}()

    output_file = "output_verbose_d$d.txt"  
    open(output_file, "a") do io                                                                                                                                                      
        write(io, "Running a simulation for dataset of states with dimensin $d \n\n")

        for catalyst in catalysts
            write(io, "The catalyst: $catalyst \n")

            plain_locc_impossible_count, catalysis_possible_count = run_catalysis_simulation(dataset_l, dataset_r, catalyst)
            
            write(io, "  Pairs with impossible plain LOCC conversion: $plain_locc_impossible_count \n")
            write(io, "  Pairs with conversion enabled by catalyst: $catalysis_possible_count \n")
            write(io, "--------------------------------------------------------- \n")

            results_dict[catalyst] = catalysis_possible_count

        end

    end

    plot_catalysis_results(results_dict)
end
