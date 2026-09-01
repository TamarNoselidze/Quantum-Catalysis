include("helper_functions.jl")

function run_dimension_analysis(dims, n_pairs, catalyst; n_population=2000, tol=1e-9)
    results = Dict{Int, Dict{String, Any}}()

    for d in dims
        println("=== Dimension d = $d ===")

        x_side, y_side = generate_iid_pairs(d, n_pairs)   
        population = generate_simplex_dataset(d, n_population)  # separate reference sample

        comparable_count = 0
        incomparable_count = 0
        catalysis_possible_count = 0

        for i in 1:n_pairs
            x = @views x_side[:, i]
            y = @views y_side[:, i]

            if is_locc_convertible(x, y; tol=tol)
                comparable_count += 1          # tracked, not discarded
            else
                incomparable_count += 1
                if is_catalysis_possible(x, y, catalyst; tol=tol) ||
                   is_catalysis_possible(y, x, catalyst; tol=tol)
                    catalysis_possible_count += 1
                end
            end
        end

        relative_volume_comparable = comparable_count / n_pairs
        catalysis_rate = incomparable_count > 0 ? catalysis_possible_count / incomparable_count : NaN

        rep_state = @views x_side[:, 1]   # example state for cone-size estimate
        fwd_frac, bwd_frac = locc_cone_fractions(rep_state, population; tol=tol)

        results[d] = Dict(
            "relative_volume_comparable"          => round(relative_volume_comparable, digits=4),
            "catalysis_rate_among_incomparable"    => round(catalysis_rate, digits=4),
            "example_forward_cone_fraction"        => round(fwd_frac, digits=4),
            "example_backward_cone_fraction"       => round(bwd_frac, digits=4),
        )

        println("  Comparable (relative volume): ", results[d]["relative_volume_comparable"])
        println("  Catalysis helps (of incomparable): ", results[d]["catalysis_rate_among_incomparable"])
        println("  Example state forward cone: ", results[d]["example_forward_cone_fraction"])
        println("  Example state backward cone: ", results[d]["example_backward_cone_fraction"])
    end

    return results
end

# Example usage:
# results = run_dimension_analysis(4:7, 2000, [0.6, 0.4])