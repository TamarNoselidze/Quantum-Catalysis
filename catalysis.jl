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
    for p in 0.5:step:1.0
    # for p in 0.6:step:0.67
        catalyst = [round(p, digits=4), round(1 - p, digits=4)]
        push!(catalysts, catalyst)
    end 
    return catalysts
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
    savefig(p, "catalysis_curve_d$dim.png")
    println("Plot saved as catalysis_curve_d$dim.png")
end


function single_dataset_analysis(d, dataset)
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]
    catalysts = sample_catalysts()

    results_dict = Dict{Vector{Float64}, Int}()
    total_count = 0
    best_catalyst = Vector{Float64}
    best_catalyst_ct = 0

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
            total_count = plain_locc_impossible_count

            if catalysis_possible_count > best_catalyst_ct
                best_catalyst_ct = catalysis_possible_count
                best_catalyst = catalyst
            end
        end
    end

    return best_catalyst, best_catalyst_ct, results_dict, total_count

end



function analyze_single_transformation(dataset)
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]
    catalysts = sample_catalysts()

    n_samples = size(dataset_l, 2)

    found_count = 0 
    
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
                println("Transformation pair index: $i")
                println("Total catalysts that worked for this pair: ", length(working_p_values))
                println("It works for p ranging from $(minimum(working_p_values)) to $(maximum(working_p_values))")
                println("----------------------------------------------------------------------\n")
                
                # stop after a few specific transformations
                # if found_count >= 3
                #     break 
                # end
            end
        end
    end
end



if abspath(PROGRAM_FILE) == @__FILE__ 
    d = 5
    @load "./datasets_40k/dataset_40k_d$d.jld2" dataset

    # best_catalyst, best_catalyst_ct, results_dict, total_count = single_dataset_analysis(d, dataset)
    # catalyst_ratio = best_catalyst_ct / total_count

    println("Dimension of the states: $d")
    # println("The best catalyst: $best_catalyst")
    # println("Catalysing ratio: $catalyst_ratio")
    # plot_catalysis_results(results_dict, d, total_count)

    analyze_single_transformation(dataset)
end
