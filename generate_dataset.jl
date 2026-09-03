using Distributions    # Standard Julia package that gives access to Dirichlet distribution + rand()
using JLD2     


function generate_simplex_dataset(dimension, num_samples)
    
    dist = Dirichlet(dimension, 1.0)
    # Constructs a Dirichlet distribution of the given dimension, with all concentration parameters equal to 1.0
    # Crucially, the symmetric Dirichlet with all αi​=1 is precisely the uniform distribution over the simplex

    
    dataset_matrix = rand(dist, num_samples)   #Draws num_samples independent samples from this Dirichlet distribution
    # rand(dist, n) returns a matrix of size (dimension, n); each column is a sampled probability vector   

    
    return dataset_matrix
end



if abspath(PROGRAM_FILE) == @__FILE__   # Only executes if you run this file directly, not when it's pulled in via include from another file
    d = 4
    n_samples = 20000

    dataset = generate_simplex_dataset(d, n_samples)

    @save "dataset_20k_d7.jld2" dataset
end

