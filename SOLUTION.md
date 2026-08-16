# Syllable Discovery in Sequential Data: Solution Writeup

**Result.** ARI **0.947** / NMI **0.941** / Hungarian-matched accuracy **0.976** on the
provided `data/data.npz` (100,000 steps, K = 10 given). The segment structure is
recovered exactly (251 predicted vs 251 true segments), with **zero errors outside
the ±half-window bands around true transitions** and 9.7 mislabeled samples per
transition, meaning each boundary is localized to about ±5 steps.

**Reproduce.** The entire solution (code, evidence, figures, and evaluation) is one
self-contained notebook, `investigation.ipynb`. It runs end to end in a couple of
minutes on a laptop, no GPU:

```bash
pip install numpy scipy scikit-learn matplotlib jupyter
jupyter nbconvert --to notebook --execute --inplace investigation.ipynb
```

(or open it in Jupyter and Run All). Inference uses only `X`, its time order, and
the given K = 10. Predictions are frozen to `outputs/` **before** ground truth is
loaded; truth is accessed once, in the evaluation section. All seeds are fixed, and
the final cell records the environment (Python 3.12.13, numpy 2.4.6, scipy 1.18.0,
scikit-learn 1.9.0) under which the numbers above reproduce exactly.

## Thought process (in notebook order)

Throughout, the disclosed generator form (origin-centered circles, each in its own
2-plane inside a shared 4-D subspace, fixed angular velocity, isotropic Gaussian
noise) is treated as a hypothesis to verify from `X`, never as a substitute for
measurement.

**1. Validate and look (§1, §2).** The data is a 100,000 × 20 matrix with no
missing values and column means close to zero. Row norms are 23.1 ± 3.0, so the
points form a thin shell around the origin rather than a filled cloud. I center
the data but do not z-score it: the 20 columns are components of one position
vector, and rescaling each column separately would distort that shared geometry.
A raw heatmap shows an oscillatory signal whose character switches in blocks of a
few hundred steps. This block scale becomes an internal correctness criterion
that needs no ground truth.

**2. PCA to measure the dimensionality (§3).** PCA answers how many of the 20
channels carry independent signal. Four eigenvalues (149, 112, 91, 64) stand
above a flat floor of sixteen values near 8.0, and the 7.8× gap between the
fourth and fifth lets a pre-registered eigengap rule retain d = 4 without any
threshold doing hidden work. The flat floor is itself informative: sixteen
directions sharing one variance is what isotropic noise looks like, and it
provides the noise estimate for free (8.00 measured, against the disclosed
2.83² = 8.01).

**3. Check that the projection preserves the shell (§3).** After projecting to
4-D the radii are 20.15 ± 3.26 with a minimum of 5.0, which is what the extreme
of 100,000 draws should look like, not an outlier. Normalizing each point to unit
length is therefore numerically stable. It discards a nearly constant radius and
with it the radial half of the noise, leaving a direction on the unit sphere S³.

**4. Baseline: pointwise k-means, judged without ground truth (§4).** Running
k-means on the denoised 4-D positions gives labels that switch 23,263 times, once
every ~4 steps, against visible blocks lasting hundreds of steps. It is rejected
on that evidence alone, and the failure is diagnostic: a position only encodes
the current *phase* of the oscillation, not the motion pattern that generated it,
so regimes whose orbits cross the same region cannot be separated sample by
sample. The label has to be computed from a stretch of trajectory, not a point.

**5. The spherical detour.** Since the data lives on a sphere, my first instinct
was to work in genuine spherical coordinates and cluster the angles. This fights
the geometry instead of using it: angle charts on S³ have a 2π wraparound and
pole singularities, so nearby points can receive distant coordinates; distances
in angle space depend on where you are on the sphere; and windowed averages of
angles would need circular statistics. More fundamentally, what identifies a
regime is not a location on the sphere but the 2-plane the trajectory sweeps, and
planes are objects of linear algebra, not of any coordinate chart. So I dropped
coordinates and worked with the algebra directly.

**6. The plane is the label, and the lap-averaged outer product represents it
(§5).** A 2-plane in 4-D has no normal vector, so a basis-free representation is
needed. The motion itself delivers one: averaging the outer product u uᵀ over at
least half a lap covers the plane's directions evenly, so the average depends
only on the plane, and the plane can be read back from the average as the set of 
directions it leaves nonzero. Isotropic noise has no preferred direction, so it 
cannot tilt this average.

**7. One time scale, measured from `X` (§5).** The averaging window must cover at
least half a lap of the slowest circle while staying inside one visit. The global
ACF first crosses zero at lag 51, roughly a quarter cycle, giving a window of 205
samples with stride 13. This lands inside the corridor [200, ~400) implied by the
disclosed scales, which validates the generator hypothesis while keeping the
pipeline's only time scale data-derived. Before building on the model I check it
per window: the top two eigenvalues of the windowed second moment carry 94.7% of
its trace at the median, so windows really do lie on great circles (the low tail
comes from windows that straddle a transition).

**8. Fingerprint, cluster, clean up (§6, §7).** The fingerprint is the vector of
16 windowed averages of uᵢuⱼ. For these, Euclidean distance equals the Frobenius
distance between plane projectors, so plain k-means (K = 10 as given, 20
restarts, keeping the best inertia, an X-only criterion) is matched to the
representation rather than a heuristic. A width-15 sliding mode filter removes
isolated flips; the width is the descriptor correlation length (window/stride,
about 16), not a tuned value. Window labels expand to samples by nearest window
center, and predictions are frozen to disk before ground truth is loaded.

## Metrics and error anatomy (§8, §9)

| variant | ARI | NMI | matched acc |
|---|---:|---:|---:|
| **fingerprint k-means + mode filter** | **0.947** | **0.941** | **0.976** |
| fingerprint k-means, no smoothing | 0.942 | 0.934 | 0.973 |
| pointwise k-means baseline | rejected X-only (23,263 switches) | | |

The geometry does the work; smoothing adds only the last few millipoints.
Accuracy on samples more than half a window away from any true transition is
**100.0000%**. Every residual error lies in the ±102-sample bands around the 250
true transitions, at 9.7 per boundary, so each boundary is localized to about ±5
samples. With 251 predicted vs 251 true segments, nothing is hallucinated and
nothing is merged. Mode-filter widths from 9 to 31 give identical metrics to four
decimal places, so the single free parameter is not a tuning knob. Internally
estimated quantities also cross-match the config the pipeline never read: noise
floor 8.00 vs 2.83² = 8.01, d = 4 vs subspace-dim 4, window 205 inside [200, 400),
mean segment 398 vs dwell 400.

## What didn't work

- **Pointwise clustering** (kept in the notebook as the baseline). It fails for
  the structural reason in step 4, and its failure mode dictated the windowed
  design.
- **Spherical coordinates.** Abandoned at the design stage for the reasons in
  step 5: wraparound, chart singularities, and a position-dependent metric make
  windowed statistics and distances unreliable, while the projector achieves the
  same phase-invariance exactly and for free.

## Known failure modes

- **Coplanar regimes.** Two regimes circling the *same* plane at different speeds
  have identical fingerprints, because the representation records where the
  trajectory circles but not how fast. The condition is detectable from `X` alone
  (two cluster centroids at near-zero distance), and the remedy is one added
  speed feature, applied only if triggered.
- **Boundary bands.** A 205-sample average cannot localize a switch more finely
  than its own support. All residual error is this limit; none of it is regime
  confusion.
- **Close planes.** Ten planes packed into 4-D leave some pairs at small
  principal angles, and cluster margins shrink with plane distance (coplanarity
  is the limiting case).

## Next steps (stopped at the time budget)

1. **Boundary refinement.** The segment skeleton is already exact, so re-estimate
   each detected transition with progressively shorter windows centered on it,
   attacking the 9.7 errors per boundary directly.
2. **Calibrated uncertainty.** Emit per-sample confidence and flag the
   ±half-window bands as ambiguous instead of forcing hard labels there.
3. **Speed feature behind the coplanarity trigger.** For example a windowed
   oriented area (signed sweep rate in the fitted plane), added only when two
   centroids nearly coincide.
