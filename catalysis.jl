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



function plot_catalysis_results(results_dict, dim, total_pairs)
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


function single_dataset_analysis(d, dataset, cat_dim)

    #For one fixed dimension d, split one big pre-generated dataset into two disjoint halves
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]
    catalysts = sample_catalysts(dimension=cat_dim)

    results_dict = Dict{Vector{Float64}, Int}()
    total_count = 0
    # best_catalyst = Vector{Float64}
    best_catalyst = Float64[]

    best_catalyst_ct = 0

    if cat_dim == 2
        output_file = "./outputs/dataset_analysis/output_verbose_d$d.txt" 
    elseif cat_dim == 3
        output_file = "./outputs/dataset_analysis/output_verbose_3Dcat_d$d.txt"
    end  

    open(output_file, "w") do io                                                                                                                                                      
        write(io, "Running a simulation for dataset of states with dimension $d \n\n")

        for catalyst in catalysts
            write(io, "The catalyst: $catalyst \n")

            plain_locc_impossible_count, catalysis_possible_count = run_catalysis_simulation(dataset_l, dataset_r, catalyst)
            
            write(io, "  Pairs with impossible plain LOCC conversion: $plain_locc_impossible_count \n")
            write(io, "  Pairs with conversion enabled by catalyst: $catalysis_possible_count \n")
            write(io, "--------------------------------------------------------- \n")

            results_dict[catalyst] = catalysis_possible_count
            total_count = plain_locc_impossible_count

            if catalysis_possible_count > best_catalyst_ct
                best_catalyst_ct = catalysis_possible_count
                best_catalyst = catalyst
            end
        end

        # Write the final summary after the analysis completes; function is unchanged besides this
        catalyst_ratio = total_count > 0 ? best_catalyst_ct / total_count : NaN
        write(io, "\n=========================================================== \n")
        write(io, "SUMMARY \n")
        write(io, "Best catalyst: $best_catalyst \n")
        write(io, "Conversions enabled by best catalyst: $best_catalyst_ct \n")
        write(io, "Total incomparable pairs: $total_count \n")
        write(io, "Catalysing ratio: $catalyst_ratio \n")

    
    end

    return best_catalyst, best_catalyst_ct, results_dict, total_count, output_file

end


#Purpose: analyze a single transformation (x, y) for all possible catalysts and track which p-values work
function analyze_single_transformation(dataset, d)
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]
    catalysts = sample_catalysts(dimension=3)

    n_samples = size(dataset_l, 2)

    found_count = 0 
    

    output_file = "./temp$d.txt"  
    open(output_file, "w") do io                                                                                                                                                      
        write(io, "Analysing transformations for dimensin $d \n\n")
    
        for i in 1:n_samples
            x = @views dataset_l[:, i]
            y = @views dataset_r[:, i]

            # find a transformation that is impossible by plain LOCC
            if !is_locc_convertible(x, y)
                
                working_p_values = Float64[]
                
                # test ALL catalysts on this ONE specific transformation (x, y)
                for catalyst in catalysts
                    if is_catalysis_possible(x, y, catalyst)
                        push!(working_p_values, catalyst[1])
                    end
                end
                
                # if this transformation can be catalyzed, track which p-values worked
                if !isempty(working_p_values)
                    found_count += 1

                    x_sorted = sort(x, rev=true)  # sorted in descending order
                    y_sorted = sort(y, rev=true)

                    write(io, "Transformation pair: $x_sorted and $y_sorted \n")
                    write(io, "Transformation pair index: $i \n")

                    write(io, "Total catalysts that worked for this pair: $(length(working_p_values)) \n")
                    write(io, "Working p-value intervals: \n")
                    intervals = p_val_intervals(working_p_values)
                    write(io, "Number of disjoint intervals: $(length(intervals)) \n")
                    for interval in intervals
                        if length(interval) == 1
                            write(io, "  -> Isolated value: $(interval[1])\n")
                        else
                            write(io, "  -> [$(interval[1]), $(interval[end])]  ($(length(interval)) values)\n")
                        end
                    end
                    write(io, "----------------------------------------------------------------------\n\n")
                    
                    # stop after a few specific transformations
                    # if found_count >= 3
                    #     break 
                    # end
                end
            end
        end
    end
end



if abspath(PROGRAM_FILE) == @__FILE__ 
    d = 12
    cat_dim = 3
    @load "./datasets_40k/dataset_40k_d$d.jld2" dataset

    best_catalyst, best_catalyst_ct, results_dict, total_count, output_file = single_dataset_analysis(d, dataset, cat_dim)
    catalyst_ratio = best_catalyst_ct / total_count

    println("Dimension of the states: $d")
    println("The best catalyst: $best_catalyst")
    println("Catalysing ratio: $catalyst_ratio")
    if cat_dim == 2
        plot_catalysis_results(results_dict, d, total_count)
    elseif cat_dim == 3
        plot_3D_catalysis_results(results_dict, d, total_count)
    end 
    
    localized_analysis(results_dict, total_count, output_file)

    # analyze_single_transformation(dataset, d)
end
