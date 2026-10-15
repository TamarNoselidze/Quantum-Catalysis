using JSON

function save_results_json(results_dict, filepath)
    # Reshape Dict{Vector{Float64}, Int} into an array of dicts for JSON
    data_to_save = [Dict("catalyst" => cat, "count" => count) for (cat, count) in results_dict]
    
    open(filepath, "w") do f
        JSON.print(f, data_to_save, 4) # The '4' adds pretty-printing indentation
    end
    println("Raw results saved to $filepath")
end

function load_results_json(filepath)
    raw_data = JSON.parsefile(filepath)
    results_dict = Dict{Vector{Float64}, Int}()
    
    # Reconstruct the original Julia dictionary
    for item in raw_data
        cat = convert(Vector{Float64}, item["catalyst"])
        count = convert(Int, item["count"])
        results_dict[cat] = count
    end
    
    println("Loaded $(length(results_dict)) catalysts from $filepath")
    return results_dict
end