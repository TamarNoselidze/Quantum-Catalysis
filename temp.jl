using Distributions

# Function to create a dataset matrix of simplex points
function generate_simplex_dataset(dimension, num_samples)
    # Dirichlet distribution 
    dist = Dirichlet(dimension, 1.0)
    
    dataset_matrix = rand(dist, num_samples)
    
    return dataset_matrix
end


d = 4
n_samples = 100

dataset = generate_simplex_dataset(d, n_samples)


# catalysts = 


# function 