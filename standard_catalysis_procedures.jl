"""
This file implements core functions for the study of catalysis in bipartite quantum systems, including:
    - Checking majorization between two Schmidt coefficient vectors
    - Checking deterministic LOCC convertibility through Nielsen's theorem
    - Computing the Schmidt coefficient vector of a tensor product |x⟩⊗|y⟩ of two Schmidt vectors
    - Applying cheap necessary-condition pre-filters to quickly rule out impossible catalysis transitions before the full check:
      1. maximally entangled states and product states never catalyze
      2. alpha_1 <= alpha_1' and alpha_n >= alpha_n' must hold
    - Determining whether a specified transition is catalyzed by a given catalyst state

Convention: Schmidt coefficients are probabilities (i.e. already squared amplitudes); thus lambda_i >= 0, sum_i lambda_i = 1.
"""

using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# 0. Validation of probability vectors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. Majorization and LOCC convertibility
# ---------------------------------------------------------------------------

function is_majorized(x, y; tol=1e-9)
    # Return True iff x is majorized by y, i.e. x ≺ y.

    x = validate_probability_vector(x)
    y = validate_probability_vector(y)

    n = max(length(x), length(y))
    x_padded = zeros(n)
    y_padded = zeros(n)

    x_padded[1:length(x)] .= x
    y_padded[1:length(y)] .= y

    x_sorted = sort(x_padded, rev=true)
    y_sorted = sort(y_padded, rev=true)

    cumsum_x = cumsum(x_sorted)
    cumsum_y = cumsum(y_sorted)

    return all(cumsum_x .<= cumsum_y .+ tol)
end

function is_locc_convertible(alpha, alpha_prime; tol=1e-9)
    # Checks LOCC convertibility between two Schmidt vectors 
    # in both directions.

    fwd = is_majorized(alpha, alpha_prime, tol=tol)
    bwd = is_majorized(alpha_prime, alpha, tol=tol)

    if fwd && bwd
        message = "alpha and alpha' are equivalent under plain LOCC"
    elseif fwd
        message = "alpha -> alpha' is possible under plain LOCC"
    elseif bwd
        message = "alpha' -> alpha is possible under plain LOCC"
    else
        message = "alpha <-> alpha' is impossible under plain LOCC"
    end

    return (forward=fwd, backward=bwd, comparable=fwd || bwd, message=message)

end


# ---------------------------------------------------------------------------
# 2. Tensor product of Schmidt vectors
# ---------------------------------------------------------------------------

function tensor_product_schmidt_vector(x, y)

    x = validate_probability_vector(x)
    y = validate_probability_vector(y)

    products =  vec([x_i * y_j for x_i in x, y_j in y])
    return sort(products, rev=true)
end

# ---------------------------------------------------------------------------
# 3. Cheap pre-filters (necessary but not sufficient)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 4. Catalysis check
# ---------------------------------------------------------------------------

function is_catalysis_possible(alpha, alpha_prime, beta; tol=1e-9, verbose=false)
    alpha = validate_probability_vector(alpha)
    alpha_prime = validate_probability_vector(alpha_prime)
    beta = validate_probability_vector(beta)

    # Step 1: plain LOCC comparability
    plain = is_locc_convertible(alpha, alpha_prime, tol=tol)
    
    if plain.forward
        return (catalyzes=false,
                reason= "alpha -> alpha' already possible under plain LOCC; catalysis is not applicable.")
    end
    if plain.backward
        return (catalyzes=false,
                reason= "alpha' -> alpha possible under plain LOCC; the forward transition alpha -> alpha' is impossible.")
    end

    # Step 2: degenerate catalyst filters
    if is_product_state(beta, tol=tol)
        return (catalyzes=false, 
                reason= "beta is a product state; cannot catalyze.")
    end
    if is_maximally_entangled(beta, tol=tol)
        return (catalyzes=false, 
                reason= "beta is a maximally entangled state; cannot catalyze.") 
    end

    # Step 3: Lemma 3 necessary condition
    if !necessary_condition_lemma3(alpha, alpha_prime, tol=tol)
        return (catalyzes=false,
                reason= "The necessary condition (alpha_1 <= alpha_1' and alpha_n >= alpha_n') failed; "
                    * "no catalyst can work for this pair.")
    end

    # Step 4: full tensor + majorization check
    joint_source = tensor_product_schmidt_vector(alpha, beta)
    joint_target = tensor_product_schmidt_vector(alpha_prime, beta)
    result = is_majorized(joint_source, joint_target, tol=tol)

    if verbose
        println("joint_source (alpha ⊗ beta): ", round.(joint_source, digits=4))
        println("joint_target (alpha'⊗ beta): ", round.(joint_target, digits=4))
    end

    if result
        return (catalyzes=true,
                reason="alpha⊗beta ≺ alpha'⊗beta holds: beta catalyzes this transition.")
    else
        return (catalyzes=false,
                reason="Passed cheap filters, but alpha⊗beta is NOT majorized by alpha'⊗beta.")
    end
end



