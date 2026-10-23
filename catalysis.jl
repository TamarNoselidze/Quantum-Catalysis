using JLD2
using Plots

include("catalysis_helper_functions.jl")
include("visualization.jl")
include("sampling.jl")
include("localised_analysis.jl")
include("save_to_json.jl")

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



function single_dataset_analysis(state_dim, dataset, catalysts, output_file)

    #For one fixed dimension d, split one big pre-generated dataset into two disjoint halves
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]

    results_dict = Dict{Vector{Float64}, Int}()
    total_count = 0
    # best_catalyst = Vector{Float64}
    best_catalyst = Float64[]

    best_catalyst_ct = 0

    open(output_file, "w") do io                                                                                                                                                      
        write(io, "Running a simulation for dataset of states with dimension $state_dim \n\n")

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
function analyze_single_transformation(dataset, state_dim, )
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]
    catalysts = sample_catalysts(dimension=4)

    n_samples = size(dataset_l, 2)

    found_count = 0 
    

    output_file = "./temp$d.txt"  
    open(output_file, "w") do io                                                                                                                                                      
        write(io, "Analysing transformations for dimension $state_dim \n\n")
    
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
                end
            end
        end
    end
end




if abspath(PROGRAM_FILE) == @__FILE__ 
    state_dim = 12
    catalyst_dim = 4
    # @load "./datasets_40k/dataset_40k_d$state_dim.jld2" dataset

    output_txt = "./outputs/dataset_analysis/4D_Catalysts/state_$(state_dim)D_catalyst_$(catalyst_dim)D.txt" 
    output_json = "./outputs/dataset_analysis/4D_Catalysts/Raw_Results/state$(state_dim)D_cat$(catalyst_dim)D.json"

    # Only run the heavy simulation if we haven't saved the JSON yet
    if isfile(output_json)
        println("Found existing results, loading from JSON...")
        results_dict = load_results_json(output_json)
        
        total_count = 20000 # hardcoded now, but has to be fixed!!
    else
        println("No JSON found, running full simulation...")
        @load "./datasets_40k/dataset_40k_d$state_dim.jld2" dataset
        catalysts = sample_catalysts(dimension=catalyst_dim)
        
        best_catalyst, best_catalyst_ct, results_dict, total_count = single_dataset_analysis(state_dim, dataset, catalysts, output_txt)
        catalyst_ratio = best_catalyst_ct / total_count
        
        save_results_json(results_dict, output_json)
    end

    # catalysts = sample_catalysts(dimension=catalyst_dim)

    # best_catalyst, best_catalyst_ct, results_dict, total_count = single_dataset_analysis(state_dim, dataset, catalysts, output_file)
    # catalyst_ratio = best_catalyst_ct / total_count

    # println("Dimension of the states: $state_dim")
    # println("The best catalyst: $best_catalyst")
    # println("Catalysing ratio: $catalyst_ratio")

    # plotlyjs()  
    # plot_catalysis_results(results_dict, state_dim, total_count, catalyst_dim)

    
    localised_4D_analysis(results_dict, total_count, output_txt)

    # analyze_single_transformation(dataset, d)
end
