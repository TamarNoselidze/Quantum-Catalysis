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
    return fwd || bwd
    
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

    x_joint = tensor_product_schmidt_vector(x, catalyst)
    y_joint = tensor_product_schmidt_vector(y, catalyst)

    result = is_locc_convertible(x_joint, y_joint)
    return result

end



