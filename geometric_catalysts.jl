# geometric_catalysts.jl
# Do the best catalysts have a geometric shape p_i ∝ r^(i-1)?
#  A. fit r to each grid winner and measure how geometric it is
#  B. sweep r for many catalyst sizes k; pick r on ORIGINAL pairs, evaluate once on NEW pairs
#  C. summary tables of r* and catalytic power versus k and d
# Everything printed to the terminal is also saved to OUT_TXT.

using JLD2, Printf
include("catalysis.jl")     # helpers (is_locc_convertible, tensor_product_schmidt_vector, ...)

const KS         = [2, 3, 4]     # catalyst sizes ; [2, 3, 4, 5, 6, 8, 12] 
const DS         = [4, 8, 12]                 # state dimensions to test
const RS         = collect(0.05:0.01:0.95)    # ratio grid
const N_ORIG     = 20_000                     # original pairs (per half)
const N_NEW      = 100_000                    # new pairs (per half), same files as the validation
const TOL        = 1e-9                       # tolerance of the majorization test (test with 1e-13)
const FLAG_BELOW = 1e-6                       # flag catalysts with a coefficient smaller than this
const OUT_TXT    = "./outputs/geometric_catalysts_tol$(TOL).txt"

# print to the terminal AND to the log file
const LOG = Ref{IO}(devnull)
out(s)   = (print(s); print(LOG[], s))
outln(s = "") = out(string(s, "\n"))

# number formatting helpers ("-" when a value does not exist)
f2(x) = isnan(x) ? "-" : @sprintf("%.2f", x)
f3(x) = isnan(x) ? "-" : @sprintf("%.3f", x)
f4(x) = isnan(x) ? "-" : @sprintf("%.4f", x)

# grid winners from the full simulations, with earlier results:
# (k, d) => (best catalyst, in-sample %, % on the new pairs from validation_results.txt)
const GRID_WINNERS = Dict(
    (2, 4)  => ([0.65, 0.35],              2.37, 2.06),
    (2, 8)  => ([0.68, 0.32],              4.26, 4.31),
    (2, 12) => ([0.70, 0.30],              4.84, 4.78),
    (3, 4)  => ([0.48, 0.32, 0.20],        3.27, 3.11),
    (3, 8)  => ([0.55, 0.29, 0.16],        6.43, 6.51),
    (3, 12) => ([0.57, 0.29, 0.14],        7.26, 7.22),
    (4, 4)  => ([0.39, 0.28, 0.20, 0.13],  4.04, 3.71),
    (4, 5)  => ([0.44, 0.27, 0.18, 0.11],  5.65, 5.64),
    (4, 6)  => ([0.43, 0.28, 0.18, 0.11],  6.94, 6.85),
    (4, 7)  => ([0.44, 0.28, 0.18, 0.10],  7.21, 7.15),
    (4, 8)  => ([0.46, 0.29, 0.16, 0.09],  7.76, 7.65),
    (4, 9)  => ([0.46, 0.28, 0.17, 0.09],  8.39, 7.93),
    (4, 10) => ([0.49, 0.27, 0.16, 0.08],  8.20, 7.97),
    (4, 11) => ([0.50, 0.28, 0.15, 0.07],  8.56, 8.28),
    (4, 12) => ([0.50, 0.27, 0.15, 0.08],  8.58, 8.47),
)

geometric_catalyst(k, r) = (w = [r^(i - 1) for i in 1:k]; w ./ sum(w))

# best-fit r for a given catalyst vector (least squares on a fine grid) + L2 distance to that fit
function fit_r(p)
    k = length(p)
    rs = 0.001:0.001:0.999
    errs = [sum((p .- geometric_catalyst(k, r)) .^ 2) for r in rs]
    i = argmin(errs)
    return rs[i], sqrt(errs[i])
end

# Same logic as run_catalysis_simulation, but with an explicit tolerance TOL.
# Returns (incomparable pairs, pairs made convertible by the catalyst).
function run_sim(left, right, catalyst; tol = TOL)
    impossible = 0
    catalyzed  = 0
    for i in 1:size(left, 2)
        x = @views left[:, i]
        y = @views right[:, i]
        if !is_locc_convertible(x, y; tol = tol)
            impossible += 1
            xj = tensor_product_schmidt_vector(x, catalyst)
            yj = tensor_product_schmidt_vector(y, catalyst)
            if is_locc_convertible(xj, yj; tol = tol)
                catalyzed += 1
            end
        end
    end
    return impossible, catalyzed
end

function load_orig(d)
    data = JLD2.load("./datasets_40k/dataset_40k_d$d.jld2", "dataset")
    return data[:, 1:N_ORIG], data[:, N_ORIG+1:2N_ORIG]
end

function load_new(d)
    data = JLD2.load("./datasets_validation/val_d$(d)_$(N_NEW)pairs.jld2", "dataset")
    return data[:, 1:N_NEW], data[:, N_NEW+1:2N_NEW]
end

# conversions enabled for each r (threaded; each catalyst is independent)
function sweep_r(left, right, k)
    cts  = zeros(Int, length(RS))
    tots = zeros(Int, length(RS))
    Threads.@threads for i in eachindex(RS)
        tots[i], cts[i] = run_sim(left, right, geometric_catalyst(k, RS[i]))
    end
    return cts, tots
end

function print_header()
    outln("GEOMETRIC CATALYSTS: do the best catalysts have a geometric shape?")
    outln("A geometric catalyst with k levels has squared Schmidt coefficients proportional to 1, r, r^2, ..., r^(k-1),")
    outln("with 0 < r < 1, so one number r describes it. Small r = peaked (little entanglement); r near 1 = flat")
    outln("(close to maximally entangled).")
    outln("Catalytic power = pairs made convertible by the catalyst / pairs plain LOCC cannot convert, in percent.")
    outln("Setup: $N_ORIG original pairs and $N_NEW new pairs per state dimension; $(length(RS)) values of r per row")
    outln("(from $(first(RS)) to $(last(RS))); majorization tolerance TOL = $TOL.")
    outln()
end

# ---------- A. shape test ----------
function shape_test()
    outln("A. SHAPE TEST: are the best catalysts from the full grid simulations geometric?")
    outln("   For each grid winner, the closest geometric catalyst was found (no simulation needed).")
    outln("   k       number of levels of the catalyst")
    outln("   d       dimension of the states")
    outln("   r_fit   ratio r of the closest geometric catalyst")
    outln("   dist    distance between the grid winner and that geometric catalyst (0 = exactly geometric;")
    outln("           below about 0.02 = close)")
    outln()
    out(@sprintf("%-3s %-3s | %-6s %-7s | %s\n", "k", "d", "r_fit", "dist", "grid winner -> closest geometric catalyst"))
    fits = Dict{Tuple{Int,Int},Tuple{Float64,Float64}}()
    for key in sort(collect(keys(GRID_WINNERS)))
        k, d = key
        w = GRID_WINNERS[key][1]
        r, dist = fit_r(w)
        fits[key] = (r, dist)
        out(@sprintf("%-3d %-3d | %-6s %-7s | %s -> %s\n",
                     k, d, f3(r), f4(dist), w, round.(geometric_catalyst(k, r), digits=3)))
    end
    outln()
    return fits
end

function run_all()
    print_header()
    fits = shape_test()
    best_rows = []   # (k, d, r, vector, orig_pct, new_pct, new_se)
    n_flagged = 0

    outln("B. Best geometric catalyst for each catalyst size k and state dimension d")
    outln("   r was chosen on the original pairs, then the catalyst was scored once on the new pairs.")
    outln("   r*          best ratio found for the grid simulation")
    outln("   r_fit       ratio implied by the grid winner's shape (from A); '-' where no grid winner exists (k > 4)")
    outln("   orig_%      catalytic power of the geometric catalyst on the original pairs (in-sample)")
    outln("   grid_%      catalytic power of the best catalyst of the full grid simulation, original pairs (in-sample)")
    outln("   new_%       catalytic power of the geometric catalyst on the new pairs (unbiased)")
    outln("   +-se        standard error of new_% (about 2 se gives a 95% range)")
    outln("   grid_new_%  the grid winner on the same new pairs (validation_results.txt), for a paired comparison")
    outln("   catalyst    squared Schmidt coefficients of the geometric catalyst; ! = smallest coefficient below $FLAG_BELOW")
    outln("   Reading it: orig_% close to grid_% and new_% close to grid_new_% means the geometric family")
    outln("   captures the best catalysts.")
    outln()
    out(@sprintf("%-3s %-3s | %-6s %-6s | %-8s %-8s | %-8s %-6s %-11s | %s\n",
                 "k", "d", "r*", "r_fit", "orig_%", "grid_%", "new_%", "±se", "grid_new_%", "catalyst"))

    for d in DS
        oL, oR = load_orig(d)
        nL, nR = load_new(d)

        for k in KS
            cts, tots = sweep_r(oL, oR, k)
            pct = 100 .* cts ./ tots

            ibest  = argmax(cts)
            r_best = RS[ibest]
            cat    = geometric_catalyst(k, r_best)

            tot_n, ct_n = run_sim(nL, nR, cat)       # new data, once
            p  = ct_n / tot_n
            se = 100 * sqrt(p * (1 - p) / tot_n)

            r_fit    = haskey(fits, (k, d)) ? fits[(k, d)][1] : NaN
            grid     = haskey(GRID_WINNERS, (k, d)) ? GRID_WINNERS[(k, d)][2] : NaN
            grid_new = haskey(GRID_WINNERS, (k, d)) ? GRID_WINNERS[(k, d)][3] : NaN
            flagged  = minimum(cat) < FLAG_BELOW
            n_flagged += flagged

            out(@sprintf("%-3d %-3d | %-6s %-6s | %-8s %-8s | %-8s %-6s %-11s | %s%s\n",
                         k, d, f2(r_best), f3(r_fit), f2(pct[ibest]), f2(grid),
                         f2(100p), f2(se), f2(grid_new), round.(cat, digits=3), flagged ? "  !" : ""))
            push!(best_rows, (k, d, r_best, cat, pct[ibest], 100p, se))
        end
    end

    # ---------- C. summary tables ----------
    outln("\nC1. BEST RATIO r* by catalyst size k (rows) and state dimension d (columns)")
    outln("    Small r = peaked catalyst, r near 1 = flat catalyst. Shows how the best shape changes with k and d.")
    outln()
    out(@sprintf("%-4s", "k")); for d in DS; out(@sprintf("%-8s", "d=$d")); end; outln()
    for k in KS
        out(@sprintf("%-4d", k))
        for d in DS
            x = first(r for r in best_rows if r[1] == k && r[2] == d)
            out(@sprintf("%-8s", f2(x[3])))
        end
        outln()
    end

    outln("\nC2. CATALYTIC POWER on the new pairs (%), best geometric catalyst, with standard error")
    outln("    Shows how much a larger catalyst (down the rows) or larger states (across the columns) helps.")
    outln()
    out(@sprintf("%-4s", "k")); for d in DS; out(@sprintf("%-16s", "d=$d")); end; outln()
    for k in KS
        out(@sprintf("%-4d", k))
        for d in DS
            x = first(r for r in best_rows if r[1] == k && r[2] == d)
            out(@sprintf("%-16s", @sprintf("%.2f ± %.2f", x[6], x[7])))
        end
        outln()
    end

    if n_flagged > 0
        outln("\nNOTE on rows marked '!' ($n_flagged rows)")
        outln("These catalysts have a coefficient below $FLAG_BELOW. Multiplied by small state coefficients, such")
        outln("entries can fall below the comparison tolerance (TOL = $TOL), which loosens the exact majorization")
        outln("test and can inflate the catalytic power. To check, rerun with TOL = 1e-13: real effects survive,")
        outln("numerical artifacts disappear.")
    end
end

function main()
    mkpath(dirname(OUT_TXT))
    open(OUT_TXT, "w") do io
        LOG[] = io
        try
            run_all()
        finally
            LOG[] = devnull
        end
    end
    println("\nSaved output to $OUT_TXT")
end

main()