#Purpose: given a list of 2D catalyst parameter values p that successfully catalyzed some specific transition,
#return a list of intervals of consecutive values
function p_val_intervals(working_p_values; step=0.01, p0=0.5)
    idx = [round(Int, (p - p0) / step) for p in working_p_values]
    intervals = Vector{Vector{Float64}}()
    current = [working_p_values[1]]
    for j in 2:length(working_p_values)
        if idx[j] == idx[j-1] + 1
            push!(current, working_p_values[j])
        else
            push!(intervals, current)
            current = [working_p_values[j]]
        end
    end
    push!(intervals, current)
    return intervals
end



#Purpose: generate a list of catalysts for a given dimension and their step size (so far only 2D and 3D)
function sample_catalysts(step=0.01; dimension=2)
    catalysts = Vector{Vector{Float64}}()
    
    if dimension == 2
        for p in 0.5:step:1.0
        # for p in 0.6:step:0.67
            catalyst = [round(p, digits=4), round(1 - p, digits=4)]
            push!(catalysts, catalyst)
        end
    
    elseif  dimension ==3
        steps_total = round(Int, 1.0 / step)
        
        # Loop through all possible values for the first two probabilities
        for i in 0:steps_total
            for j in 0:(steps_total - i)
                # The third probability is whatever is left over
                k = steps_total - i - j
                
                # only want descending order
                if i >= j >= k
                    # Convert back to standard decimal probabilities and round
                    p1 = round(i * step, digits=4)
                    p2 = round(j * step, digits=4)
                    p3 = round(k * step, digits=4)
                    
                    push!(catalysts, [p1, p2, p3])
                end
            end
        end

    elseif dimension == 4
        steps_total = round(Int, 1.0 / step)
        
        # Loop through the first three probabilities
        for i in 0:steps_total
            for j in 0:(steps_total - i)
                for k in 0:(steps_total - i - j)
                    # The fourth probability is the remainder
                    l = steps_total - i - j - k
                    
                    # Enforce strictly descending order for 4D
                    if i >= j >= k >= l
                        p1 = round(i * step, digits=4)
                        p2 = round(j * step, digits=4)
                        p3 = round(k * step, digits=4)
                        p4 = round(l * step, digits=4)
                        
                        push!(catalysts, [p1, p2, p3, p4])
                    end
                end
            end
        end
    end

    return catalysts
end