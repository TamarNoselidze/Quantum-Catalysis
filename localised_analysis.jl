#Purpose: perform a localized analysis of the 3D catalysis results, identifying 
#the best catalysts near specific boundaries of the parameter space
function localised_3D_analysis(results_dict, total_count, output_file)
    local_best_catalyst = Float64[]
    local_best_ct = 0
    best_aab_cat = Float64[]
    best_aab_ct = 0

    best_abb_cat = Float64[]
    best_abb_ct = 0

    for (cat, count) in results_dict
        # Define "close to the line" as p3 being 0.03 or less
        if cat[3] <= 0.03 
            if count > local_best_ct
                local_best_ct = count
                local_best_catalyst = cat
            end
        end

        # Test for (a, a, b) where p1 == p2
        if isapprox(cat[1], cat[2], atol=1e-4)
            if count > best_aab_ct
                best_aab_ct = count
                best_aab_cat = cat
            end
        end
        
        # Test for (a, b, b) where p2 == p3
        if isapprox(cat[2], cat[3], atol=1e-4)
            if count > best_abb_ct
                best_abb_ct = count
                best_abb_cat = cat
            end
        end
    end

    local_ratio = local_best_ct / total_count
    
    ratio_aab = best_aab_ct / total_count
    ratio_abb = best_abb_ct / total_count

    #added details to the text output file; function is unchanged besides this
    open(output_file, "a") do io   # "a" = append, so this adds on after the existing SUMMARY block
        write(io, "\n=========================================================== \n")
        write(io, "LOCALIZED ANALYSIS \n")
        write(io, "Local max near edge (p3<=0.03): $local_best_catalyst, ratio: $local_ratio \n")
        write(io, "Local max on (a,a,b) border: $best_aab_cat, ratio: $ratio_aab \n")
        write(io, "Local max on (a,b,b) border: $best_abb_cat, ratio: $ratio_abb \n")
    end
    
    # println("Local max near p1+p2=1: $local_best_catalyst")
    # println("Local catalysing ratio: $local_ratio")

    println("Local max on (a,a,b) border: $best_aab_cat")
    println("Local (a,a,b) ratio: $ratio_aab")
    
    println("Local max on (a,b,b) border: $best_abb_cat")
    println("Local (a,b,b) ratio: $ratio_abb")
end




function localised_4D_analysis(results_dict, total_count, output_file)
    local_best_catalyst = Float64[]
    local_best_ct = 0
    
    # (a,a,b,c)
    best_aabc_cat = Float64[]
    best_aabc_ct = 0
    
    # (a,b,b,c)
    best_abbc_cat = Float64[]
    best_abbc_ct = 0
    
    # (a,b,c,c)
    best_abcc_cat = Float64[]
    best_abcc_ct = 0

    # (a,a,a,b)
    best_aaab_cat = Float64[]
    best_aaab_ct = 0

    # (a,a,b,b)
    best_aabb_cat = Float64[]
    best_aabb_ct = 0

    # (a,b,b,b)
    best_abbb_cat = Float64[]
    best_abbb_ct = 0


    # (pq,p(1-q),(1-p)q,(1-p)(1-q))
    best_pq_cat = Float64[]
    best_pq_ct = 0

    # (p², p(1-p), p(1-p), (1-p)²)
    best_pq2_cat = Float64[]
    best_pq2_ct = 0


    for (cat, count) in results_dict
        # Define "close to the line" as p4 being 0.05 or less
        if cat[4] <= 0.03 
            if count > local_best_ct
                local_best_ct = count
                local_best_catalyst = cat
            end
        end

        # Test for (a,a,b,c) where p1 == p2
        if isapprox(cat[1], cat[2], atol=1e-4)
            if count > best_aabc_ct
                best_aabc_ct = count
                best_aabc_cat = cat
            end
        end
        
        # Test for (a,b,b,c) where p2 == p3
        if isapprox(cat[2], cat[3], atol=1e-4)
            if count > best_abbc_ct
                best_abbc_ct = count
                best_abbc_cat = cat
            end
        end

        # Test for (a,b,c,c) where p3 == p4
        if isapprox(cat[3], cat[4], atol=1e-4)
            if count > best_abcc_ct
                best_abcc_ct = count
                best_abcc_cat = cat
            end
        end

        ##  "doubly degenerate cases"
        # Test for (a,a,a,b) where p1 == p2 == p3
        if isapprox(cat[1], cat[2], atol=1e-4) && isapprox(cat[2], cat[3], atol=1e-4)            
            if count > best_aaab_ct
                best_aaab_ct = count
                best_aaab_cat = cat
            end
        end

        # Test for (a,a,b,b) where p1 == p2 and p3 == p4
        if isapprox(cat[1], cat[2], atol=1e-4) && isapprox(cat[3], cat[4], atol=1e-4)
            if count > best_aabb_ct
                best_aabb_ct = count
                best_aabb_cat = cat
            end
        end

        # Test for (a,b,b,b) where p2 == p3 == p4
        if isapprox(cat[2], cat[3], atol=1e-4) && isapprox(cat[3], cat[4], atol=1e-4)
            if count > best_abbb_ct
                best_abbb_ct = count
                best_abbb_cat = cat
            end
        end

        ## PQ Params
        # Test for (pq,p(1-q),(1-p)q,(1-p)(1-q))
        if isapprox(cat[1] * cat[4], cat[2] * cat[3], atol=1e-4)
            if count > best_pq_ct
                best_pq_ct = count
                best_pq_cat = cat
            end
        end

        if isapprox(cat[1] * cat[4], cat[2] * cat[3], atol=1e-4) && isapprox(cat[2], cat[3], atol=1e-4)
            if count > best_pq2_ct
                best_pq2_ct = count
                best_pq2_cat = cat
            end
        end

    end

    local_ratio = local_best_ct / total_count
    
    # ratio_aabc = best_aabc_ct / total_count
    # ratio_abbc = best_abbc_ct / total_count
    # ratio_abcc = best_abcc_ct / total_count
    # ratio_aaab = best_aaab_ct / total_count
    # ratio_aabb = best_aabb_ct / total_count
    # ratio_abbb = best_abbb_ct / total_count
    # ratio_pq   = best_pq_ct / total_count
    # ratio_pq2  = best_pq2_ct / total_count

    ratios = [
        "(a,a,b,c)" => (best_aabc_cat, best_aabc_ct / total_count),
        "(a,b,b,c)" => (best_abbc_cat, best_abbc_ct / total_count),
        "(a,b,c,c)" => (best_abcc_cat, best_abcc_ct / total_count),
        "(a,a,a,b)" => (best_aaab_cat, best_aaab_ct / total_count),
        "(a,a,b,b)" => (best_aabb_cat, best_aabb_ct / total_count),
        "(a,b,b,b)" => (best_abbb_cat, best_abbb_ct / total_count),
        "(pq,p(1-q),(1-p)q,(1-p)(1-q))" => (best_pq_cat, best_pq_ct / total_count),
        "(p², p(1-p), p(1-p), (1-p)²)"  => (best_pq2_cat, best_pq2_ct / total_count)
    ]

    open(output_file, "a") do io   
        write(io, "\n=========================================================== \n")
        write(io, "LOCALISED ANALYSIS \n")
        write(io, "Local max near p4 <= 0.03: $local_best_catalyst  |  Ratio: $local_ratio \n\n")
        
        # Loop over the ratios array and print results
        for (family_name, (cat_vector, ratio_val)) in ratios
            if isempty(cat_vector)
                write(io, "$family_name : No catalysts sampled in this family.\n")
            else
                write(io, "$family_name : Best is $cat_vector  |  Ratio: $ratio_val \n")
            end
        end
        write(io, "=========================================================== \n")
    end    
end
