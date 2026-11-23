# For a chosen catalyst size k and state dimension d:
#  1. Set an initial (previosuly found as best performing) geometric catalyst c0 = c(r*)
#  2. Generate an epsilon-neighborhood in the simplex with step size delta
#  3. Run LOCC simulations across all neighbors on the original 20k state pairs
#  4. Determine whether c0 is a strict local maximum, an isolated peak, or a flat plateau
#  5. Measure non-geometric deviation (d_geom) for the best-performing neighbors

using JLD2, Printf, LinearAlgebra, Statistics

include("catalysis_helper_functions.jl")
include("geometric_catalysts.jl")
include("printing.jl")



# Standard optimal geometric ratios r* from geometric_catalysts.jl (original 20k sweeps)
const OPTIMAL_R = Dict(
    # --- 3D catalysts ---
    # (3, 4)  => 0.65,
    # (3, 5)  => 0.62,
    # (3, 6)  => 0.59,
    # (3, 7)  => 0.56,
    # (3, 8)  => 0.54,
    # (3, 9)  => 0.53,
    # (3, 10) => 0.51,
    # (3, 11) => 0.50,
    # (3, 12) => 0.49,
    (3, 20) => 0.49,
    (3, 30) => 0.49,
    (3, 40) => 0.45,
    (3, 50) => 0.45,

    # --- 4D catalysts ---
    # (4, 4)  => 0.70,
    # (4, 5)  => 0.63,
    # (4, 6)  => 0.64,
    # (4, 7)  => 0.62,
    # (4, 8)  => 0.59,
    # (4, 9)  => 0.59,
    # (4, 10) => 0.56,
    # (4, 11) => 0.54,
    # (4, 12) => 0.54,

    (4, 20) => 0.51,
    (4, 30) => 0.49,
    (4, 40) => 0.52,
    (4, 50) => 0.48,

  
)



# Generate candidates within Euclidean ball ||c - c0||_2 <= epsilon with step delta
function sample_local_neighborhood(c0, epsilon, step)
    k = length(c0)
    neighbors = Vector{Vector{Float64}}()

    push!(neighbors, copy(c0))  # including the initial c0 itselfas well

    if k == 3
        p1_min = max(1.0/3.0, c0[1] - epsilon) # the shifting will not change the sorted, descending order of the coefficients
        p1_max = min(1.0,     c0[1] + epsilon)
        p2_min = max(0.0,     c0[2] - epsilon)
        p2_max = min(1.0,     c0[2] + epsilon)

        for p1 in p1_min:step:p1_max
            for p2 in p2_min:step:p2_max
                p3 = 1.0 - p1 - p2
                if p1 >= p2 >= p3 >= 0.0
                    candidate = [round(p1, digits=5), round(p2, digits=5), round(p3, digits=5)]
                    if norm(candidate - c0) <= epsilon && candidate != c0
                        push!(neighbors, candidate)
                    end
                end
            end
        end

    elseif k == 4
        p1_min = max(0.25, c0[1] - epsilon)
        p1_max = min(1.0,  c0[1] + epsilon)
        p2_min = max(0.0,  c0[2] - epsilon)
        p2_max = min(1.0,  c0[2] + epsilon)
        p3_min = max(0.0,  c0[3] - epsilon)
        p3_max = min(1.0,  c0[3] + epsilon)

        for p1 in p1_min:step:p1_max
            for p2 in p2_min:step:p2_max
                for p3 in p3_min:step:p3_max
                    p4 = 1.0 - p1 - p2 - p3
                    if p1 >= p2 >= p3 >= p4 >= 0.0
                        candidate = [round(p1, digits=5), round(p2, digits=5), round(p3, digits=5), round(p4, digits=5)]
                        if norm(candidate - c0) <= epsilon && candidate != c0
                            push!(neighbors, candidate)
                        end
                    end
                end
            end
        end
    else
        error("Dimension k = $k not implemented in local sampler.")
    end

    # Remove any potential duplicates from rounding
    unique!(neighbors)

    return neighbors
end



# Extract the incomparable state pairs from the dataset once to accelerate simulation. -- should also implement this in other files!
function extract_incomparable_pairs(dataset, n_pairs; tol=1e-9)
    left  = dataset[:, 1:n_pairs]
    right = dataset[:, n_pairs+1:2n_pairs]
    
    incomp_x = Vector{Vector{Float64}}()
    incomp_y = Vector{Vector{Float64}}()
    
    for i in 1:n_pairs
        x = left[:, i]
        y = right[:, i]
        if !is_locc_convertible(x, y; tol=tol)
            push!(incomp_x, x)
            push!(incomp_y, y)
        end
    end
    
    return incomp_x, incomp_y
end



# Count conversions enabled by a catalyst on the pre-extracted incomparable pairs.
function count_conversions(incomp_x, incomp_y, catalyst; tol=1e-9)
    n_incomp = length(incomp_x)
    conversions = 0
    for i in 1:n_incomp
        x = incomp_x[i]
        y = incomp_y[i]
        xj = tensor_product_schmidt_vector(x, catalyst)
        yj = tensor_product_schmidt_vector(y, catalyst)

        if is_locc_convertible(xj, yj; tol=tol)
            conversions += 1
        end
    end
    return conversions
end



function run_neighborhood_pilot(state_dim, cat_dim, r_star; epsilon=0.03, step=0.002, n_pairs=20_000, tol=1e-9)
    # Anchor geometric catalyst
    c0 = round.(geometric_catalyst(cat_dim, r_star), digits=5)
    
    # exctract incomparable pairs
    dataset_path = "./datasets_40k/dataset_40k_d$(state_dim).jld2"
    data = JLD2.load(dataset_path, "dataset")
    incomp_x, incomp_y = extract_incomparable_pairs(data, n_pairs; tol=tol)
    
    # sampe neighborhood
    neighbors = sample_local_neighborhood(c0, epsilon, step)
    
    conversions = [count_conversions(incomp_x, incomp_y, cat; tol=tol) for cat in neighbors]
    
    return (
        state_dim    = state_dim,
        cat_dim      = cat_dim,
        r_star       = r_star,
        c0           = c0,
        epsilon      = epsilon,
        step         = step,
        n_pairs      = n_pairs,
        total_incomp = length(incomp_x),
        neighbors    = neighbors,
        conversions  = conversions
    )
end





function main()
    target_k = 4          #3 for 3D catalysts only, 4 for 4D only, or nothing for both
    epsilon  = 0.03

    # Filter keys by target_k if set
    all_keys    = collect(keys(OPTIMAL_R))
    active_keys = (target_k === nothing) ? all_keys : filter(x -> x[1] == target_k, all_keys)
    sorted_keys = sort(active_keys, by = x -> (x[1], x[2]))
    total_runs  = length(sorted_keys)

    for (idx, (cat_dim, state_dim)) in enumerate(sorted_keys)
        r_star = OPTIMAL_R[(cat_dim, state_dim)]
        
        # use step = 0.002 for k=3, and step = 0.005 for k=4
        step = (cat_dim == 3) ? 0.002 : 0.005
        
        out_file = "./outputs/neighborhood_analysis/k$(cat_dim)_d$(state_dim).txt"
        
        println("\n" * "="^75)
        @printf("[%d/%d] Running neighborhood analysis: k=%d, d=%d (r*=%.3f, step=%.4f)...\n",
                idx, total_runs, cat_dim, state_dim, r_star, step)
        println("="^75)
        
        results = run_neighborhood_pilot(state_dim, cat_dim, r_star; epsilon=epsilon, step=step)
        print_neighborhood_summary(results; output_file=out_file)
    end
    
    println("\nAll neighborhood analyses completed successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end