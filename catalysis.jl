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


run_catalysis_simulation(dataset, catalyst)
