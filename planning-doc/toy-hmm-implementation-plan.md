# Toy HMM Implementation Plan

Step-by-step build plan for the toy Hidden Markov Model (forward-backward, then Baum-Welch).
Follows the supervisor's advice: implement first, learn the theory from the code.

## Step 0. Fix the toy model on paper first

Pick a model small enough that you can check numbers by hand.

- 3 hidden states: S1, S2, S3.
- 4 observation symbols: a, b, c, d.
- π: length-3 vector, e.g. (0.6, 0.3, 0.1). Must sum to 1. It only answers where does the chain start. 
- A: 3×3 transition matrix. Each row sums to 1. Make it clearly non-uniform. How to make the matrix non-uniform?
  (e.g. S1 mostly stays in S1, S2 tends to move to S3).
- B: 3×4 emission matrix. Each row sums to 1. Each state prefers a different symbol
  so states are distinguishable.

Write these down as fixed constants. Do not learn them yet. Steps 1–3 rely on
known parameters so that you know what the right answers should be.

## Step 1. Generate synthetic sequences from the model

Write a sampler that walks the chain:

1. Draw the first state from π. This is the start of the chain.
2. Draw an observation from that state's row of B.
3. Draw the next state from that state's row of A.
4. Repeat for T steps.

Keep both the hidden state path and the observation sequence.

- One short sequence (T ≈ 5) for hand checks.
- One long sequence (T ≈ 500–1000) for learning later.

Keep the true state path aside as the answer key.

## Step 2. Forward algorithm (α)

$\alpha_t(i)$ = probability of seeing $o_1 \dots o_t$ and being in state $i$ at time $t$.

- Initialise: $\alpha_1(i) = \pi_i \, B[i, o_1]$
- Recurse: $\alpha_t(j) = \left( \sum_i \alpha_{t-1}(i) \, A[i, j] \right) B[j, o_t]$
- Terminate: $P(O \mid \lambda) = \sum_i \alpha_T(i)$

Store $\alpha$ as a $T \times N$ matrix.

**Check (most valuable test):** on the T=5 sequence, compute $P(O \mid \lambda)$ by brute force,
enumerating all $3^5 = 243$ state paths and summing the $\pi \cdot A \cdot B$ products along each path.
It must equal the forward result to floating-point precision.

## Step 3. Backward algorithm (β)

$\beta_t(i)$ = probability of the future observations $o_{t+1} \dots o_T$ given state $i$ at time $t$.

- Initialise: $\beta_T(i) = 1$ for all $i$
- Recurse backwards: $\beta_t(i) = \sum_j A[i, j] \, B[j, o_{t+1}] \, \beta_{t+1}(j)$


Store $\beta$ as a $T \times N$ matrix.

**Checks:**
- $P(O \mid \lambda) = \sum_i \pi_i \, B[i, o_1] \, \beta_1(i)$. Must match the forward result.
- For every $t$: $\sum_i \alpha_t(i) \, \beta_t(i)$ gives the same $P(O \mid \lambda)$.

If both hold, $\alpha$ and $\beta$ are correct. This is the "two ways, same number" test.

## Step 4. Scaling, before anything else

On the long sequence $\alpha$ underflows to zero (product of hundreds of numbers < 1).
Fix this now, because Baum-Welch needs long sequences.

Standard approach:
- At each $t$, divide $\alpha_t$ by its sum $c_t$ and store $c_t$.
- Scale $\beta_t$ with the same $c_t$.
- $\log P(O \mid \lambda) = \sum_t \log c_t$.

Re-run the Step 2 and 3 checks in log space.
Alternative: do everything with log-sum-exp (cleaner, slower to write).

## Step 5. Posteriors γ and ξ (E-step)

Both come directly from $\alpha$ and $\beta$.

- $\gamma_t(i) = \dfrac{\alpha_t(i) \, \beta_t(i)}{\sum_k \alpha_t(k) \, \beta_t(k)}$
  Probability of being in state $i$ at time $t$.
  Shape $T \times N$ ($T$ time steps, $N$ hidden states). Each row sums to 1.
- $\xi_t(i, j) = \dfrac{\alpha_t(i) \, A[i, j] \, B[j, o_{t+1}] \, \beta_{t+1}(j)}{P(O \mid \lambda)}$
  Probability of the transition $i \to j$ between $t$ and $t+1$.
  Shape $(T-1) \times N \times N$. Each $t$-slice sums to 1.

**Checks:**
- $\sum_j \xi_t(i, j) = \gamma_t(i)$
- $\arg\max$ of $\gamma_t$ on synthetic data should mostly agree with the true state path from Step 1.

## Step 6. Baum-Welch re-estimation (M-step)

Given $\gamma$ and $\xi$, the new parameters are normalised expected counts:

- $\pi_i \leftarrow \gamma_1(i)$
- $A[i, j] \leftarrow \dfrac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$
  (expected transitions $i \to j$ over expected visits to $i$)
- $B[j, k] \leftarrow \dfrac{\sum_{t : o_t = k} \gamma_t(j)}{\sum_t \gamma_t(j)}$
  (expected emissions of symbol $k$ from state $j$ over expected visits to $j$)

Loop: E-step (Steps 2–5) then M-step, until $\log P(O \mid \lambda)$ stops increasing.

**Check:** log-likelihood must never decrease between iterations.
If it does, the bug is almost always in $\xi$ or the scaling.

## Step 7. Recover the true parameters

Start from random $\pi, A, B$ and run Baum-Welch on the long synthetic sequence.
Compare the learned $A$ and $B$ with the true ones from Step 0.

Expected behaviour (not bugs):
- States may come out permuted. Match them by matching emission rows.
- With a single sequence some parameters may be poorly identified.
  Use several random restarts and keep the best likelihood.

## Step 8. Extensions that matter for the real data

Only after Step 7 works, in this order:

1. **Multiple sequences.** Accumulate numerators and denominators of Step 6 across
   sequences before dividing. Needed regardless of the ordering strategy chosen for the cells.
2. **Viterbi decoding.** Single most likely state path. Useful for interpretation
   and a cross-check against $\arg\max \gamma$.
3. **Continuous emissions.** Replace the discrete table $B$ with e.g. a Gaussian per state.
   Real cells are expression vectors, not symbols. Only the M-step for $B$ changes;
   $\alpha, \beta, \gamma, \xi$ stay identical.

## Suggested code layout

- One module with pure functions: `sample`, `forward`, `backward`, `posteriors`,
  `baum_welch_step`, `fit`.
- One test file with the brute-force check and the $\alpha \cdot \beta$ consistency check.
- Parameters as plain NumPy arrays so the toy and the real data share the same code.

## Time budget

- Sitting 1: Steps 0–4.
- Sitting 2: Steps 5–7.
- Step 8 after the Thursday meeting, once the sequence-construction decision is made.


# Study Note: How to Define Zones for Sampling

## Rule

To draw randomly from a row of probabilities, turn the row into zones on the stretch 0 to 1:

1. Pick the row you are drawing from: π for the first state, the current state's row of A for the next state, the current state's row of B for the symbol.
2. Start at 0.
3. Add the row's values one by one, left to right. Each running total is a cut point.
4. Assign zones in order: first option owns 0 → first total, second owns first total → second total, and the last one ends exactly at 1.0.
5. Throw the dart: get a random number r between 0 and 1. The zone that contains r is the pick.

Check: the last cut point must be exactly 1.0. If not, the row does not sum to 1 and is wrong.

Constraint that matters: each zone's width must equal its probability. The running-total recipe guarantees this automatically. The order of zones is convention; the widths are sacred.

## Analogy

The row of probabilities is a dartboard. Each option owns a slice, and the slice's size is its probability. The stick is 0 to 1 because that is where darts (random numbers) land — zones outside 0 to 1 would never be hit. Rows sum to 1 so the slices exactly fill the board: no gap, no overlap, every dart has exactly one home. The dart r is not a probability — it is one landing spot. The widths are the probabilities.

## Worked example

Row: A from S2 = (0.1, 0.3, 0.6).

Running totals: 0.1, then 0.1 + 0.3 = 0.4, then 0.4 + 0.6 = 1.0. ✓ ends at 1.0.

Zones:

```
S1: 0 → 0.1
S2: 0.1 → 0.4
S3: 0.4 → 1.0
```

Dart r = 0.83. Check zone by zone: between 0 and 0.1? No. Between 0.1 and 0.4? No. Between 0.4 and 1.0? Yes → pick S3.

A different row gives different zones. B in S2 = (0.1, 0.7, 0.1, 0.1) gives a: 0 → 0.1, b: 0.1 → 0.8, c: 0.8 → 0.9, d: 0.9 → 1.0. Same recipe, different fences. Dart r = 0.44 lands in b's zone → pick b.

## Project connection

This is how the toy HMM sampler produces synthetic data: every draw in the T = 5 hand run and the long T = 500 sequence is this zone trick applied to one row — π once at the start, then A's current row for each move and B's current row for each symbol. The synthetic sequences later become the test data for forward-backward and Baum-Welch. In the real scRNA-seq model, nature does this drawing itself: a cell in a hidden state "throws the dart" to pick its next state and its expression signal.

## You are done when

- You can build the zones for any row of π, A, or B without help.
- You can say why the last cut point must be 1.0.
- You can say why r = 0.83 with zones (0.1, 0.4, 1.0) picks S3.
- You can explain the difference between a zone width (a probability) and r (one landing spot).

---

Note this settles the hanging question too: **r = 0.83 → S3 is the state at t = 2.** Table so far:

| t | r for state | state | r for symbol | observation |
|---|---|---|---|---|
| 1 | 0.72 | S2 | 0.44 | b |
| 2 | 0.83 | S3 | | |

Next dart from the list is r = **0.29**, for t = 2's observation. We're in S3, so B in S3 zones apply: a: 0→0.1, b: 0.1→0.2, c: 0.2→0.6, d: 0.6→1.0. Which symbol?