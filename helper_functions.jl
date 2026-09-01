include("generate_dataset.jl")

function validate_probability_vector(x; tol=1e-9, sum_tol=1e-6)
    x = float.(x)

    if any(x .< -tol)
        throw(ArgumentError("Probability vector must be nonnegative."))
    end
    if abs(sum(x) - 1) > sum_tol
        throw(ArgumentError("Probability vector must sum to 1."))
    end

    return x
end

function is_majorized(x, y; tol=1e-9)
    # Return True if x is majorized by y

    n = max(length(x), length(y))
    x_padded = zeros(n)
    y_padded = zeros(n)

    x_padded[1:length(x)] .= x   # .= for elementwise adding 
    y_padded[1:length(y)] .= y

    x_sorted = sort(x_padded, rev=true)  # sorted in descending order
    y_sorted = sort(y_padded, rev=true)

    cumsum_x = cumsum(x_sorted)
    cumsum_y = cumsum(y_sorted)

    return all(cumsum_x .<= cumsum_y .+ tol)
end


function is_locc_convertible(x, y; tol=1e-9)
    # Checks LOCC convertibility between two Schmidt vectors in both directions.

    fwd = is_majorized(x, y, tol=tol)
    bwd = is_majorized(y, x, tol=tol)

    # return (forward=fwd, backward=bwd, comparable=fwd || bwd)  # could be useful later, for now we just wanna know if converstion is possible
    return fwd || bwd #returns true if pair is comparable in either direction 
    
end


function tensor_product_schmidt_vector(x, y)

    # x = validate_probability_vector(x)
    # y = validate_probability_vector(y)

    products =  vec([x_i * y_j for x_i in x, y_j in y])
    return sort(products, rev=true)
end



function is_maximally_entangled(beta; tol=1e-9)
    beta = validate_probability_vector(beta)
    n = length(beta)
    uniform_val = 1.0 / n
    return all(abs.(beta .- uniform_val) .< tol)
end



function is_product_state(beta; tol=1e-9)
    beta = validate_probability_vector(beta)
    return any(abs.(beta .- 1.0) .< tol)
end



function necessary_condition_lemma3(alpha, alpha_prime; tol=1e-9)
    alpha = validate_probability_vector(alpha)
    alpha_prime = validate_probability_vector(alpha_prime)

    n = max(length(alpha), length(alpha_prime))
    a = zeros(n)
    ap = zeros(n)
    
    a[1:length(alpha)] .= alpha
    ap[1:length(alpha_prime)] .= alpha_prime

    a_sorted = sort(a, rev=true)
    ap_sorted = sort(ap, rev=true)

    lemma3_largest = a_sorted[1] <= ap_sorted[1] + tol
    lemma3_smallest = a_sorted[end] >= ap_sorted[end] - tol

    return lemma3_largest && lemma3_smallest
end



function is_catalysis_possible(x, y, catalyst; tol=1e-9)

    if is_majorized(x, y, tol=tol) || is_majorized(y, x, tol=tol)
        return "States are not incomparable"
    end

    x_joint = tensor_product_schmidt_vector(x, catalyst)
    y_joint = tensor_product_schmidt_vector(y, catalyst)

    result = is_majorized(x_joint, y_joint)
    return result

end

"""
Generate n_pairs i.i.d. pairs (x_i, y_i), each independently drawn from the
uniform distribution on the d-dimensional simplex A. Avoids the problem that
S X S is not uniformly iid in A X A, since pairs share no points with each other.
"""
function generate_iid_pairs(dimension, n_pairs)
    pop = generate_simplex_dataset(dimension, 2 * n_pairs)  # 2N fresh iid draws
    x_side = pop[:, 1:2:end]   # odd-indexed columns  -> N states
    y_side = pop[:, 2:2:end]   # even-indexed columns -> N states
    return x_side, y_side
end

"""
Estimate, for a fixed state x, the fraction of `population` states that:
  - x can directly reach via LOCC (x majorized by y)   -> forward_fraction
  - can directly reach x via LOCC (y majorized by x)   -> backward_fraction
"""
function locc_cone_fractions(x, population; tol=1e-9)
    n_pop = size(population, 2) # returns size of dimension 2 of the population matrix, i.e., number of columns (sampled states)
    forward_count = 0
    backward_count = 0

    for i in 1:n_pop
        y = @views population[:, i]
        is_majorized(x, y; tol=tol) && (forward_count += 1)
        is_majorized(y, x; tol=tol) && (backward_count += 1)
    end

    return forward_count / n_pop, backward_count / n_pop
end