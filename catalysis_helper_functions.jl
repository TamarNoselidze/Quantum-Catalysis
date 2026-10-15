include("generate_dataset.jl")

function tensor_product_schmidt_vector(x, y)

    # x = validate_probability_vector(x)
    # y = validate_probability_vector(y)

    products =  vec([x_i * y_j for x_i in x, y_j in y])
    return sort(products, rev=true)
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



function is_catalysis_possible(x, y, catalyst; tol=1e-9)

    x_joint = tensor_product_schmidt_vector(x, catalyst)
    y_joint = tensor_product_schmidt_vector(y, catalyst)

    result = is_locc_convertible(x_joint, y_joint)
    return result

end