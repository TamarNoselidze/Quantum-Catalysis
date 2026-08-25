"""
Implements the core objects from Jonathan & Plenio (1999)
    - Majorization order on Schmidt coefficient vectors
    - Nielsen's theorem (dleeterministic LOCC convertibility)
    - Catalysis (ELQCC) check, with cheap necessary-condition pre-filters
      (Lemma 1: maximally entangled states never catalyze;
       Lemma 3: alpha_1 <= alpha_1' and alpha_n >= alpha_n' are necessary)
    - Monte Carlo estimator for "catalytic power" P(beta) = measure of the
      set of transitions (alpha, alpha') that beta catalyzes, under the
      uniform measure on the probability simplex.

Convention: Schmidt coefficients are probabilities (i.e. already squared
amplitudes): lambda_i >= 0, sum_i lambda_i = 1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. majorization
# ---------------------------------------------------------------------------

def is_majorized(x, y, tol=1e-9):
    """
    Return True iff x is majorized by y, i.e. x ≺ y.

    Definition: sort both vectors in decreasing order, zero-pad to equal
    length, and require that every partial sum of x is <= the corresponding
    partial sum of y:
        sum_{i=1}^k x_i <= sum_{i=1}^k y_i    for all k

    Parameters
    ----------
    x, y : array-like
        Probability vectors (nonnegative, should sum to 1). Need not be
        pre-sorted or pre-padded -- this function handles both.
    tol : float
        Numerical tolerance for the inequality and for the sum-to-one check.

    Returns
    -------
    bool
    """
    x = np.asarray(x, dtype=float) #defensive coercion - converts whatever was passed into a float NumPy array
    y = np.asarray(y, dtype=float)

    if np.any(x < -tol) or np.any(y < -tol):
        raise ValueError("Probability vectors must be nonnegative.")
    if abs(x.sum() - 1.0) > 1e-6 or abs(y.sum() - 1.0) > 1e-6:
        raise ValueError("Probability vectors must sum to 1.")

    n = max(len(x), len(y))
    x_padded = np.zeros(n)
    y_padded = np.zeros(n)
    x_padded[:len(x)] = x
    y_padded[:len(y)] = y

    x_sorted = np.sort(x_padded)[::-1]
    y_sorted = np.sort(y_padded)[::-1]

    cumsum_x = np.cumsum(x_sorted)
    cumsum_y = np.cumsum(y_sorted)

    return bool(np.all(cumsum_x <= cumsum_y + tol))


def nielsen_convertible(alpha, alpha_prime, tol=1e-9):
    """
    Check plain-LOCC convertibility both directions via Nielsen's theorem.

    Returns
    -------
    dict with keys:
        'forward'  : True if alpha -> alpha' is possible under plain LOCC
        'backward' : True if alpha' -> alpha is possible under plain LOCC
        'comparable' : True if either direction holds (i.e. NOT incomparable)
    """
    fwd = is_majorized(alpha, alpha_prime, tol=tol)
    bwd = is_majorized(alpha_prime, alpha, tol=tol)
    return {"forward": fwd, "backward": bwd, "comparable": fwd or bwd}


# ---------------------------------------------------------------------------
# 2. Tensor product of Schmidt vectors
# ---------------------------------------------------------------------------

def tensor_sorted(x, y):
    """
    Compute the Schmidt coefficient vector of |x> (x) |y>, i.e. all pairwise
    products x_i * y_j, sorted in decreasing order.

    This is the joint Schmidt vector Nielsen's theorem must be applied to
    when checking catalysis -- note it must be *re-sorted*, since the
    pairwise products of two sorted vectors are not sorted in general.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    products = np.outer(x, y).flatten()
    # np.outer(x, y) computes the outer product: given x of length m and y of length n, it returns an mxn matrix
    # where entry (i,j) is x[i] * y[j]. 
    # .flatten() then reshapes this mxn matrix into a 1D array of length m*n
    return np.sort(products)[::-1]


# ---------------------------------------------------------------------------
# 3. Cheap pre-filters (Lemmas 1 and 3)
# ---------------------------------------------------------------------------

def is_maximally_entangled(beta, tol=1e-9):
    """
    Lemma 1 filter: a maximally entangled catalyst (uniform vector) never
    catalyzes anything. 
    """
    beta = np.asarray(beta, dtype=float)
    n = len(beta)
    uniform_val = 1.0 / n
    return bool(np.all(np.abs(beta - uniform_val) < tol))


def is_product_state(beta, tol=1e-9):
    """
    A product-state catalyst (one coefficient = 1, rest = 0) is trivial:
    tensoring with it changes nothing, so it can never enable a new
    transition either. 
    """
    beta = np.asarray(beta, dtype=float)
    return bool(np.any(np.abs(beta - 1.0) < tol))


def necessary_condition_lemma3(alpha, alpha_prime, tol=1e-9):
    """
    Lemma 3 (Jonathan & Plenio): for n x n states, a NECESSARY (not
    sufficient) condition for alpha -> alpha' to be catalyzable by some
    catalyst is:
        alpha_1 <= alpha_1'      
        alpha_n >= alpha_n'      

    where both vectors are sorted decreasingly and zero-padded to equal
    length n. 

    Returns True if the necessary condition holds (search may proceed),
    False if catalysis is provably impossible for this pair.
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha_prime = np.asarray(alpha_prime, dtype=float)

    n = max(len(alpha), len(alpha_prime))
    a = np.zeros(n)
    ap = np.zeros(n)
    a[:len(alpha)] = alpha
    ap[:len(alpha_prime)] = alpha_prime

    a_sorted = np.sort(a)[::-1]
    ap_sorted = np.sort(ap)[::-1]

    largest_ok = a_sorted[0] <= ap_sorted[0] + tol
    smallest_ok = a_sorted[-1] >= ap_sorted[-1] - tol

    return bool(largest_ok and smallest_ok)


# ---------------------------------------------------------------------------
# 4. Full catalysis check pipeline
# ---------------------------------------------------------------------------

def catalyzes(alpha, alpha_prime, beta, tol=1e-9, verbose=False):
    """
    Determine whether catalyst `beta` catalyzes the transition
    alpha -> alpha'

    Pipeline (cheapest checks first):
      1. Check plain Nielsen convertibility. If alpha -> alpha' already
         works without help, then this is not a "true" catalyzed
         transition. 
         If alpha' -> alpha works instead, catalysis in the
         alpha -> alpha' direction is PROVABLY IMPOSSIBLE for any
         catalyst.
      2. Reject degenerate catalysts (product / maximally entangled).
      3. Apply Lemma 3's necessary condition; skip expensive check if it
         fails.
      4. Only then: compute the joint tensor vectors and check
         majorization directly.

    Returns
    -------
    dict with keys:
        'catalyzes' : bool
        'reason'    : str, human-readable explanation of the outcome
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha_prime = np.asarray(alpha_prime, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # Step 1: plain LOCC comparability
    plain = nielsen_convertible(alpha, alpha_prime, tol=tol)
    if plain["forward"]:
        return {"catalyzes": False,
                "reason": "alpha -> alpha' is already possible under plain LOCC, so "
                          "this is not a true catalyzed transition."}
    if plain["backward"]:
        return {"catalyzes": False,
                "reason": "alpha' -> alpha is already possible under plain LOCC, so "
                          "alpha -> alpha' can never be catalyzed by any catalyst."}

    # Step 2: degenerate catalyst filters (Lemma 1, trivial product case)
    if is_product_state(beta, tol=tol):
        return {"catalyzes": False,
                "reason": "beta is a product state and thus cannot catalyze."}
    if is_maximally_entangled(beta, tol=tol):
        return {"catalyzes": False,
                "reason": "beta is maximally entangled and thus cannot catalyze."}

    # Step 3: Lemma 3 necessary condition
    if not necessary_condition_lemma3(alpha, alpha_prime, tol=tol):
        return {"catalyzes": False,
                "reason": "Necessary condition (alpha_1 <= alpha_1', "
                          "alpha_n >= alpha_n') failed."}

    # Step 4: full tensor + majorization check
    joint_source = tensor_sorted(alpha, beta)
    joint_target = tensor_sorted(alpha_prime, beta)
    result = is_majorized(joint_source, joint_target, tol=tol)

    if verbose: #if the method caller set verbose=True, print the joint source and target vectors rounded to 4 decimal places
        print("joint_source (alpha ⊗ beta):", np.round(joint_source, 4))
        print("joint_target (alpha'⊗ beta):", np.round(joint_target, 4))

    if result:
        return {"catalyzes": True,
                "reason": "alpha⊗beta ≺ alpha'⊗beta holds; "
                          "beta catalyzes this transition."}
    else:
        return {"catalyzes": False,
                "reason": "Passed filters, but alpha⊗beta is NOT majorized by alpha'⊗beta;"
                          "beta does not catalyze this transition."}


# ---------------------------------------------------------------------------
# 5. Uniform sampling on the probability simplex
# ---------------------------------------------------------------------------

def sample_simplex(n, size=1, rng=None):
    """
    Draw `size` points (that is, `size` vectors) at random from the probability simplex
    Delta_n = {x in R^n : x_i >= 0, sum x_i = 1}.

    Returns
    np.ndarray (n-dimensional array) of shape (size, n) if size > 1, else shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    samples = rng.exponential(scale=1.0, size=(size, n))
    samples = samples / samples.sum(axis=1, keepdims=True)
    return samples[0] if size == 1 else samples


# ---------------------------------------------------------------------------
# 6. Monte Carlo estimator for catalytic power P(beta)
# ---------------------------------------------------------------------------

def estimate_catalytic_power(beta, n, num_samples=10000, rng=None, tol=1e-9):
    """
    Estimate P(beta) = mu(T(beta)), the (uniform-measure) fraction of
    randomly sampled transitions (alpha, alpha') in Delta_n x Delta_n that
    beta catalyzes.

    Parameters
    ----------
    beta : array-like
        The candidate catalyst's Schmidt vector.
    n : int
        Dimension of the source/target Schmidt vectors to sample (i.e. we
        sample alpha, alpha' uniformly from Delta_n).
    num_samples : int
        Number of Monte Carlo samples N.
    rng : np.random.Generator, optional

    Returns
    -------
    dict with:
        'P_hat'          : float, the Monte Carlo estimate of P(beta)
        'num_catalyzed'  : int, raw count of catalyzed pairs found
        'num_samples'    : int
        'std_error'      : float, standard error of the estimate
                            (binomial: sqrt(p_hat*(1-p_hat)/N))
    """
    if rng is None:
        rng = np.random.default_rng()

    alphas = sample_simplex(n, size=num_samples, rng=rng)
    alpha_primes = sample_simplex(n, size=num_samples, rng=rng)

    count = 0
    for i in range(num_samples):
        result = catalyzes(alphas[i], alpha_primes[i], beta, tol=tol)
        if result["catalyzes"]:
            count += 1

    p_hat = count / num_samples
    std_error = np.sqrt(p_hat * (1 - p_hat) / num_samples)

    return {
        "P_hat": p_hat,
        "num_catalyzed": count,
        "num_samples": num_samples,
        "std_error": std_error,
    }

