"""
    Dirichlet distribution is the standard, correct way to 
    sample Schmidt-coefficient-like vectors uniformly, since
    naively sampling each coordinate uniform on [0,1] and 
    renormalizing by the sum does not give a uniform 
    distribution on the simplex, but rather one that biases
    toward the center (maximally entangled state).
    
    Given the parameter vector alpha of dimension k, each
    sample from a Dirichlet distribution is a different 
    probability vector p = (p1, ..., pk). The coefficients
    p_i of that vector give the probabilities of the k 
    possible outcomes in a subsequent categorical experiment.  
    
""" 

using Distributions    # Standard Julia package that gives access to Dirichlet distribution + rand()
using JLD2     # Native file format for saving/loading Julia data structures to disk

# Function to create a dataset matrix of simplex points
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
    n_samples = 10000

    dataset = generate_simplex_dataset(d, n_samples)

    @save "simplex_dataset_big.jld2" dataset
    # Serializes the dataset variable to disk so it can be reloaded later without regenerating
end

