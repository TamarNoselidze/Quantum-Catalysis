function print_neighborhood_summary(res; output_file=nothing)
    neighbors    = res.neighbors
    conversions  = res.conversions
    total_incomp = res.total_incomp
    c0           = res.c0
    n_neighbors  = length(neighbors)

    c0_idx = findfirst(==(c0), neighbors)
    c0_ct  = c0_idx !== nothing ? conversions[c0_idx] : 0
    c0_pct = 100.0 * c0_ct / total_incomp

    max_ct  = maximum(conversions)
    min_ct  = minimum(conversions)
    avg_ct  = mean(conversions)
    avg_pct = 100.0 * avg_ct / total_incomp

    better_indices = findall(>(c0_ct), conversions)
    equal_indices  = findall(==(c0_ct), conversions)
    worse_indices  = findall(<(c0_ct), conversions)

    best_idx  = argmax(conversions)
    best_cat  = neighbors[best_idx]
    best_pct  = 100.0 * max_ct / total_incomp
    best_dist = norm(best_cat - c0)
    _, best_d_geom = fit_r(best_cat)   # from geometric_catalysts.jl

    buf = IOBuffer()
    println(buf, "="^75)
    println(buf, "PILOT NEIGHBORHOOD ANALYSIS REPORT")
    println(buf, "="^75)
    @printf(buf, "State dim: %d | Catalyst dim: %d | Incomparable pairs: %d\n", res.state_dim, res.cat_dim, total_incomp)
    @printf(buf, "Anchor c0: %s (r*=%.3f) | Radius ε: %.3f | Step δ: %.4f\n", string(c0), res.r_star, res.epsilon, res.step)
    @printf(buf, "Total neighbors tested: %d\n\n", n_neighbors)

    println(buf, "--- SUMMARY METRICS ---")
    @printf(buf, "Anchor c0 conversions:      %d / %d  (%.2f%%)\n", c0_ct, total_incomp, c0_pct)
    @printf(buf, "Maximum conversions found:  %d / %d  (%.2f%%)\n", max_ct, total_incomp, best_pct)
    @printf(buf, "Minimum conversions found:  %d / %d  (%.2f%%)\n", min_ct, total_incomp, 100.0 * min_ct / total_incomp)
    @printf(buf, "Average neighborhood power: %.2f%%\n\n", avg_pct)

    println(buf, "--- LOCAL PEAK STATUS ---")
    if isempty(better_indices)
        if length(equal_indices) == 1
            println(buf, ">> STATUS: STRICT LOCAL PEAK! No neighbor achieved higher or equal conversions.")
        else
            @printf(buf, ">> STATUS: LOCAL PLATEAU! %d neighbors tied with c0 (including c0 itself), none exceeded it.\n",
                    length(equal_indices))
        end
    else
        @printf(buf, ">> STATUS: NOT A PEAK. Found %d neighbor(s) that outperformed c0 (gain: +%d conversions, +%.2f%%).\n",
                length(better_indices), max_ct - c0_ct, best_pct - c0_pct)
    end
    @printf(buf, "Neighbors strictly worse:  %d (%.1f%%)\n", length(worse_indices), 100.0 * length(worse_indices) / n_neighbors)
    @printf(buf, "Neighbors equal to c0:     %d (%.1f%%)\n", length(equal_indices), 100.0 * length(equal_indices) / n_neighbors)
    @printf(buf, "Neighbors strictly better: %d (%.1f%%)\n\n", length(better_indices), 100.0 * length(better_indices) / n_neighbors)

    println(buf, "--- BEST NEIGHBOR DETAILS ---")
    @printf(buf, "Best Catalyst Vector:  %s\n", string(best_cat))
    @printf(buf, "Conversions Enabled:   %d (%.2f%%)\n", max_ct, best_pct)
    @printf(buf, "Distance to Anchor c0: %.4f\n", best_dist)
    @printf(buf, "Distance to Geometric: %.4f  (0 = purely geometric, >0 = non-geometric)\n\n", best_d_geom)

    # Top 10 ranking table
    sorted_order = sortperm(conversions, rev=true)
    println(buf, "--- TOP 10 NEIGHBORS IN B_ε(c0) ---")
    @printf(buf, "%-4s | %-10s | %-8s | %-8s | %-10s | %s\n",
            "Rank", "Conversions", "Power (%)", "Dist(c0)", "Dist(geom)", "Catalyst Vector")
    for r in 1:min(10, n_neighbors)
        idx = sorted_order[r]
        cat = neighbors[idx]
        ct  = conversions[idx]
        pct = 100.0 * ct / total_incomp
        d_c0 = norm(cat - c0)
        _, d_g = fit_r(cat)
        marker = (cat == c0) ? " [ANCHOR c0]" : ""
        @printf(buf, "%-4d | %-10d | %-7.2f%% | %-8.4f | %-10.4f | %s%s\n",
                r, ct, pct, d_c0, d_g, string(cat), marker)
    end
    println(buf, "="^75)

    report_str = String(take!(buf))
    print(report_str)

    if output_file !== nothing
        mkpath(dirname(output_file))
        write(output_file, report_str)
        println("\nSaved report to $output_file")
    end
end