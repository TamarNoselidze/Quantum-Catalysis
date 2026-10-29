using Random, Distributions, JLD2, Printf, Statistics, Plots


include("catalysis_helper_functions.jl")
include("visualization.jl")
include("sampling.jl")
include("localised_analysis.jl")
include("save_to_json.jl")

function run_catalysis_simulation(left_side, right_side, catalyst)
    plain_locc_impossible_count = 0
    catalysis_possible_count = 0

    n_samples = size(left_side, 2) # returns size of dimension 2 of the dataset matrix, i.e., number of columns (sampled states)

    for i in 1:n_samples
        # @views avoids creating an array copy of column i of dataset; it creates a lightweight reference into the matrix instead
        x = @views left_side[:, i]    # 'view all row indices from column i of dataset'
        y = @views right_side[:, i]

        if !is_locc_convertible(x, y) 
            plain_locc_impossible_count += 1

            # testing for catalysis in both directions        
            if is_catalysis_possible(x, y, catalyst)
                catalysis_possible_count += 1
            end
                    
        end 
    end 

    # println("  Pairs with impossible plain LOCC conversion: ", plain_locc_impossible_count)
    # println("  Pairs with conversion enabled by catalyst: ", catalysis_possible_count)
    return plain_locc_impossible_count, catalysis_possible_count
end



function single_dataset_analysis(state_dim, dataset, catalysts, output_file)

    #For one fixed dimension d, split one big pre-generated dataset into two disjoint halves
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]

    results_dict = Dict{Vector{Float64}, Int}()
    total_count = 0
    # best_catalyst = Vector{Float64}
    best_catalyst = Float64[]

    best_catalyst_ct = 0

    open(output_file, "w") do io                                                                                                                                                      
        write(io, "Running a simulation for dataset of states with dimension $state_dim \n\n")

        for catalyst in catalysts
            write(io, "The catalyst: $catalyst \n")

            plain_locc_impossible_count, catalysis_possible_count = run_catalysis_simulation(dataset_l, dataset_r, catalyst)
            
            write(io, "  Pairs with impossible plain LOCC conversion: $plain_locc_impossible_count \n")
            write(io, "  Pairs with conversion enabled by catalyst: $catalysis_possible_count \n")
            write(io, "--------------------------------------------------------- \n")

            results_dict[catalyst] = catalysis_possible_count
            total_count = plain_locc_impossible_count

            if catalysis_possible_count > best_catalyst_ct
                best_catalyst_ct = catalysis_possible_count
                best_catalyst = catalyst
            end
        end

        # Write the final summary after the analysis completes; function is unchanged besides this
        catalyst_ratio = total_count > 0 ? best_catalyst_ct / total_count : NaN
        write(io, "\n=========================================================== \n")
        write(io, "SUMMARY \n")
        write(io, "Best catalyst: $best_catalyst \n")
        write(io, "Conversions enabled by best catalyst: $best_catalyst_ct \n")
        write(io, "Total incomparable pairs: $total_count \n")
        write(io, "Catalysing ratio: $catalyst_ratio \n")

    
    end

    return best_catalyst, best_catalyst_ct, results_dict, total_count, output_file

end


#Purpose: analyze a single transformation (x, y) for all possible catalysts and track which p-values work
function analyze_single_transformation(dataset, state_dim)
    dataset_l = dataset[:, 1:20000] 
    dataset_r = dataset[:, 20001:40000]
    catalysts = sample_catalysts(dimension=2)

    n_samples = size(dataset_l, 2)

    found_count = 0 
    

    output_file = "./temp$state_dim.txt"  
    open(output_file, "w") do io                                                                                                                                                      
        write(io, "Analysing transformations for dimension $state_dim \n\n")
    
        for i in 1:n_samples
            x = @views dataset_l[:, i]
            y = @views dataset_r[:, i]

            # find a transformation that is impossible by plain LOCC
            if !is_locc_convertible(x, y)
                
                working_p_values = Float64[]
                
                # test ALL catalysts on this ONE specific transformation (x, y)
                for catalyst in catalysts
                    if is_catalysis_possible(x, y, catalyst)
                        push!(working_p_values, catalyst[1])
                    end
                end
                
                # if this transformation can be catalyzed, track which p-values worked
                if !isempty(working_p_values)
                    found_count += 1

                    x_sorted = sort(x, rev=true)  # sorted in descending order
                    y_sorted = sort(y, rev=true)

                    write(io, "Transformation pair: $x_sorted and $y_sorted \n")
                    write(io, "Transformation pair index: $i \n")

                    write(io, "Total catalysts that worked for this pair: $(length(working_p_values)) \n")
                    write(io, "Working p-value intervals: \n")
                    intervals = p_val_intervals(working_p_values)
                    write(io, "Number of disjoint intervals: $(length(intervals)) \n")
                    for interval in intervals
                        if length(interval) == 1
                            write(io, "  -> Isolated value: $(interval[1])\n")
                        else
                            write(io, "  -> [$(interval[1]), $(interval[end])]  ($(length(interval)) values)\n")
                        end
                    end
                    write(io, "----------------------------------------------------------------------\n\n")
                end
            end
        end
    end
end


# Number of incomparable pairs in the dataset (the denominator of every ratio)
function count_incomparable_pairs(dataset)
    half = size(dataset, 2) ÷ 2
    left  = dataset[:, 1:half]
    right = dataset[:, half+1:2half]
    return count(i -> !is_locc_convertible(view(left, :, i), view(right, :, i)), 1:half)
end

#main file should only run when no argument is given
# run with julia --project=. catalysis.jl
if abspath(PROGRAM_FILE) == abspath(@__FILE__) && isempty(ARGS)
    state_dim = 12
    catalyst_dim = 4
    # @load "./datasets_40k/dataset_40k_d$state_dim.jld2" dataset

    output_txt   = "./outputs/dataset_analysis/4D_Catalysts/state_$(state_dim)D_catalyst_$(catalyst_dim)D.txt"
    output_json  = "./outputs/dataset_analysis/4D_Catalysts/Raw_Results/state$(state_dim)D_cat$(catalyst_dim)D.json"
    output_local = replace(output_txt, ".txt" => "_localised.txt")

    mkpath(dirname(output_txt))      # create folders if missing
    mkpath(dirname(output_json))
    mkpath("./plots")

    @load "./datasets_40k/dataset_40k_d$state_dim.jld2" dataset

    # Only run the heavy simulation if we haven't saved the JSON yet
    if isfile(output_json)
        println("Found existing results, loading from JSON...")
        results_dict = load_results_json(output_json)
        total_count  = count_incomparable_pairs(dataset)

    else
        println("No JSON found, running full simulation...")
        catalysts = sample_catalysts(dimension=catalyst_dim)

        _, _, results_dict, total_count, _ = single_dataset_analysis(state_dim, dataset, catalysts, output_txt)
        
        save_results_json(results_dict, output_json)
    end

    isfile(output_local) && rm(output_local)   # avoid appending duplicate blocks on reruns
    localised_4D_analysis(results_dict, total_count, output_local)

    # catalysts = sample_catalysts(dimension=catalyst_dim)

    # best_catalyst, best_catalyst_ct, results_dict, total_count = single_dataset_analysis(state_dim, dataset, catalysts, output_file)
    # catalyst_ratio = best_catalyst_ct / total_count

    # println("Dimension of the states: $state_dim")
    # println("The best catalyst: $best_catalyst")
    # println("Catalysing ratio: $catalyst_ratio")

    # plotlyjs()  
    # plot_catalysis_results(results_dict, state_dim, total_count, catalyst_dim)

    
    # localised_4D_analysis(results_dict, total_count, output_txt)

    # analyze_single_transformation(dataset, d)
end



# ---------- Table summarizing all simulations ----------
# run with:  julia --project=. catalysis.jl table
function parse_summary(path)
    state_dim = nothing; best = nothing; cnt = nothing; total = nothing
    for ln in readlines(path)
        if startswith(ln, "Running a simulation")
            m = match(r"dimension\s+(\d+)", ln)
            m === nothing || (state_dim = parse(Int, m.captures[1]))
        elseif startswith(ln, "Best catalyst:")
            body = replace(split(ln, ":", limit=2)[2], r"[\[\]]" => "")
            best = [parse(Float64, strip(s)) for s in split(body, ",")]
        elseif startswith(ln, "Conversions enabled by best catalyst:")
            cnt = parse(Int, strip(split(ln, ":")[2]))
        elseif startswith(ln, "Total incomparable pairs:")
            total = parse(Int, strip(split(ln, ":")[2]))
        end
    end
    any(isnothing, (state_dim, best, cnt, total)) && return nothing
    return (state_dim=state_dim, cat_dim=length(best), best=best, count=cnt,
            total=total, ratio = total > 0 ? cnt / total : NaN, file=path)
end


function print_results_table()
    rows = []
    for (root, _, files) in walkdir("./outputs/dataset_analysis")
        for f in files
            endswith(f, ".txt") || continue
            r = parse_summary(joinpath(root, f))
            r === nothing || push!(rows, r)
        end
    end
    sort!(rows, by = r -> (r.cat_dim, r.state_dim))

    buf = IOBuffer()    # build the report here, then print it and save the same text
    println(buf, "SUMMARY OF ALL SIMULATIONS: best catalyst for each catalyst size and state dimension")
    println(buf)

    for r in rows
        println(buf, rpad(r.cat_dim, 9), rpad(r.state_dim, 11), rpad(r.total, 14),
                rpad(r.count, 12), rpad(round(100 * r.ratio, digits=2), 9), r.best)
    end

    report = String(take!(buf))
    print(report)                                   # terminal
    mkpath(dirname("./outputs/results_summary_table.txt"))
    write("./outputs/results_summary_table.txt", report)                      # same text, saved to file
    println("\nSaved ./outputs/results_summary_table.txt")
end


# ---Validation of the best catalysts on new independent pairs
# run with:  julia --project=. catalysis.jl validate
const N_PAIRS        = 100_000                       # new pairs per state dimension
const VAL_DIR        = "./datasets_validation"
const VALIDATION_TXT = "./outputs/validation_results.txt"

# (catalyst_dim, state_dim) => (best catalyst, original conversions, original incomparable pairs)
const WINNERS = Dict(
    (2, 4)  => ([0.65, 0.35],              173,  7311),
    (2, 8)  => ([0.68, 0.32],              473, 11091),
    (2, 12) => ([0.70, 0.30],              614, 12697),
    (3, 4)  => ([0.48, 0.32, 0.20],        239,  7311),
    (3, 8)  => ([0.55, 0.29, 0.16],        713, 11091),
    (3, 12) => ([0.57, 0.29, 0.14],        922, 12697),
    (4, 4)  => ([0.39, 0.28, 0.20, 0.13],  295,  7311),
    (4, 5)  => ([0.44, 0.27, 0.18, 0.11],  495,  8754),
    (4, 6)  => ([0.43, 0.28, 0.18, 0.11],  676,  9741),
    (4, 7)  => ([0.44, 0.28, 0.18, 0.10],  762, 10574),
    (4, 8)  => ([0.46, 0.29, 0.16, 0.09],  861, 11091),
    (4, 9)  => ([0.46, 0.28, 0.17, 0.09],  976, 11628),
    (4, 10) => ([0.49, 0.27, 0.16, 0.08],  991, 12089),
    (4, 11) => ([0.50, 0.28, 0.15, 0.07], 1057, 12350),
    (4, 12) => ([0.50, 0.27, 0.15, 0.08], 1089, 12697),
)

# New dataset for dimension d: generated once with a fixed seed, then cached on disk.
function validation_dataset(d)
    path = joinpath(VAL_DIR, "val_d$(d)_$(N_PAIRS)pairs.jld2")
    if isfile(path)
        return JLD2.load(path, "dataset")
    end
    Random.seed!(987_000 + d)                          # different seed per d, reproducible
    dataset = generate_simplex_dataset(d, 2 * N_PAIRS)
    JLD2.jldsave(path; dataset)
    return dataset
end

function validate_catalysts()
    mkpath(VAL_DIR)
    mkpath(dirname(VALIDATION_TXT))

    buf  = IOBuffer()      # build the report here, then print it and save the same text
    rows = []              # (catalyst_dim, state_dim, diff)

    println(buf, "VALIDATION OF THE BEST CATALYSTS ON NEW PAIRS")
    println(buf)
    println(buf, "Columns:")
    println(buf, "  cat     Schmidt coefficients of the catalyst")
    println(buf, "  d       dimension of the states being converted")
    println(buf, "  orig_%  catalytic power in the original sweep (where the catalyst was chosen)")
    println(buf, "  new_%   catalytic power on the new pairs (unbiased estimate)")
    println(buf, "  +-se    standard error of the percentage to its left (about 2 se gives a 95% range)")
    println(buf, "  diff    new_% minus orig_%, in percentage points (negative = original was optimistic)")
    println(buf, "  z       diff divided by its combined standard error; |z| below 2 is consistent with chance")
    println(buf)
    @printf(buf, "%-4s %-4s | %-8s %-7s | %-8s %-7s | %-7s %-6s\n",
            "cat", "d", "orig_%", "±se", "new_%", "±se", "diff", "z")

    for key in sort(collect(keys(WINNERS)))
        cat_dim, d = key
        catalyst, old_ct, old_tot = WINNERS[key]

        data  = validation_dataset(d)
        left  = data[:, 1:N_PAIRS]
        right = data[:, N_PAIRS+1:2N_PAIRS]

        tot, ct = run_catalysis_simulation(left, right, catalyst)

        p_new  = ct / tot
        se_new = sqrt(p_new * (1 - p_new) / tot)
        p_old  = old_ct / old_tot
        se_old = sqrt(p_old * (1 - p_old) / old_tot)
        diff   = 100 * (p_new - p_old)
        z      = (p_new - p_old) / sqrt(se_new^2 + se_old^2)

        @printf(buf, "%-4d %-4d | %-8.2f %-7.2f | %-8.2f %-7.2f | %-+7.2f %-+6.2f\n",
                cat_dim, d, 100p_old, 100se_old, 100p_new, 100se_new, diff, z)
        push!(rows, (cat_dim, d, diff))
    end

    println(buf, "\nMean shift (new_% - orig_%) by catalyst size, in percentage points.")
    println(buf, "Negative values are the selection bias: the original sweeps were slightly optimistic.")
    for c in (2, 3, 4)
        ds = [r[3] for r in rows if r[1] == c]
        @printf(buf, "  %d-level catalysts: %+.2f  (averaged over %d dimensions)\n", c, mean(ds), length(ds))
    end

    report = String(take!(buf))
    print(report)                       # terminal
    write(VALIDATION_TXT, report)       # same text, saved to file
    println("\nSaved $VALIDATION_TXT")
end


# --entry points (only when this file is run directly) 
if abspath(PROGRAM_FILE) == abspath(@__FILE__) && "table" in ARGS
    print_results_table()
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && "validate" in ARGS
    validate_catalysts()
end