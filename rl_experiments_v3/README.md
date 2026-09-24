# RL checkpoints - lineage v3 (167-dim state: banked trick points)

A directory separate from `rl_experiments_v2/`, created when
`coinche/rl_agent.py`'s `encode_state` gained 2 scalars (`STATE_DIM` 165 ->
167): card points already banked by my side / the opponent in the donne's
complete tricks (`_points_so_far`), normalized by a reference total (162).
Previously absent from the state (remarques_rl.md), even though it's a
direct signal for the PPO critic (close to the very quantity it regresses)
-- includes neither belote nor the last-trick bonus (cf. `_points_so_far`'s
comment), just the raw card points of tricks already won.

This input-dimension change **breaks compatibility** of every checkpoint in
`rl_experiments/` and `rl_experiments_v2/` (`fc1`'s shape changes, `size
mismatch` on load) -- hence the new lineage, the same principle as
`rl_experiments_v2/` when the scoring rule 5 had changed.

`imit.pt` regenerated here (`pretrain.py --games 5000 --epochs 20 --seed 0`,
same settings as the previous lineage): 92.8% imitation accuracy of
`HeuristicPlayer` (vs an unrecorded figure for the old `imit.pt`, not
directly comparable).

## First test: the v2 lineage's winning PPO config (`lr=3e-5, epochs=8`)

Same protocol as `rl_experiments_v2` (`--opponent heuristic` alone, 50k
episodes, same hyperparameters) from this new `imit.pt`, to test whether
the new feature helps.

| | eval_avg(3000) | win_rate | delta vs (its own) imit.pt |
|---|---|---|---|
| heuristic (reference) | +7.38 | 51.8% | -- |
| imit.pt (v3, with the feature) | +1.28 | 50.4% | -- |
| **exp_ppo_lr3e-5** (v3, with the feature) | **+20.41** | **54.6%** | **+19.13** |
| *for reference: exp_ppo_round1_lr3e-5 (v2, without the feature)* | *+17.60* | *53.8%* | *+20.68* |

**Cautious reading**: in raw value (the figure tracked throughout), it's
better (+20.41 vs +17.60). But `imit.pt` itself is also better with the
feature (+1.28 vs -3.08 before) -- probably because the same info also
helps supervised imitation of the heuristic, not just PPO fine-tuning. In
delta relative to its own starting point (the metric used everywhere else
in this session), it's actually slightly **below** (+19.13 vs +20.68) --
and this gap (~1.5 pt) is in any case well below the empirical 95% CI
(~+/-7-8 pts, one run each). No solid evidence of a net gain at this stage:
a well-motivated intuition, a verified implementation, a result compatible
with "helps a little" or "neutral", not enough to settle yet. Would need
several seeds (same remark as for `rl_experiments_v2/README.md`'s `--lr`
ablation) to conclude.

## Continuation to 100k episodes

`exp_ppo_lr3e-5/ppo_50000.pt` extended by 50k more episodes
(`--load`+`--load-value` on this same checkpoint to resume both policy AND
critic, `--episode-offset 50000`, `--seed 21` -- same hyperparameters
otherwise), to see whether the 50k performance was a plateau or an early
snapshot (cf. `exp_growing_pool`, which had kept progressing up to ~300k).
Comparison below at 5000 games (instead of 3000, for a bit more precision):

| | eval_avg(5000) | win_rate |
|---|---|---|
| heuristic (reference) | +8.07 | 51.7% |
| imit.pt | +1.91 | 50.4% |
| ppo_50000 (remeasured at n=5000) | +17.90 | 53.9% |
| **ppo_100000** | **+19.98** | **54.2%** |

(The `heuristic`/`imit.pt`/`ppo_50000` figures differ slightly from the
previous section -- measured at n=3000 vs n=5000, different donne samples,
not a real change.)

**Reading**: a delta of +2.08 between 100k and 50k. With an empirical
standard deviation of ~228-230 pts/donne, the 95% CI at n=5000 is on the
order of +/-6 points -- this delta stays within that, not distinguishable
from noise. Also to be taken with the same caveat as everywhere else this
session: the 50k->100k segment has a different training seed (21 vs 20)
from the first segment, so "more episodes" and "different seed" are
confounded, a single run can't separate them. **Conclusion: stable, no net
progress detected between 50k and 100k** -- consistent with a plateau
reached rather than a clear tendency to keep improving, but not
statistically proven for now.

PPO training doesn't save the Adam optimizer's state (m/v moments) between
two segments -- each resume (`--load`/`--load-value` in a new process)
restarts Adam "cold" on already-trained weights. Estimated impact small (a
few hundred steps disrupted out of this segment's ~100k) and already
tolerated in the segmented REINFORCE lineage; see `PPOPolicy.save()`'s
docstring (`train_ppo.py`) for detail.

## REINFORCE reference on this lineage (167-dim state)

Two REINFORCE runs (`train.py`, `CardNet`/`NeuralPolicy`) from this same
`imit.pt`, 100k episodes each, `--opponent heuristic` (no pool/PFSP),
constant `entropy-beta=0.002` (visible in the logs) -- the same two `lr`
values as the PPO ablation (`rl_experiments_v2/README.md`), to directly
test `remarques_rl.md` point 5's methodological caveat: REINFORCE's
`lr=1e-4` had never been revalidated after the scoring rule-5 fix.

## Full comparison (`eval_policy.py`, 5000 games, seed=42)

| Checkpoint | eval_avg(5000) | win_rate |
|---|---|---|
| **REINFORCE lr=3e-5, 100k** | **+25.14** | **55.3%** |
| REINFORCE lr=1e-4, 100k | +21.07 | 54.3% |
| PPO lr=3e-5, 100k | +19.98 | 54.2% |
| PPO lr=3e-5, 50k | +17.90 | 53.9% |
| heuristic | +8.07 | 51.7% |
| imit.pt | +1.91 | 50.4% |

**Central finding: REINFORCE regains the lead once treated fairly.**
`rl_experiments_v2`'s "PPO beats REINFORCE" conclusion (+17.60 PPO vs
+15.22 `seg_50000`) rested on a REINFORCE whose lr (1e-4) had never been
revalidated after rule 5 -- exactly the caveat already documented at the
time. Here, under equal treatment (same state, same lr tested on both
sides, same 100k episodes), REINFORCE `lr=3e-5` (+25.14) clearly beats PPO
`lr=3e-5` 100k (+19.98); even REINFORCE `lr=1e-4` (+21.07), already above
PPO, confirms that lowering the lr also helps REINFORCE (a delta of +4.07
relative to its own `lr=1e-4`, the same direction as the PPO ablation --
but this gap also stays under the empirical 95% CI at n=5000, ~+/-6 pts:
suggestive, not proven, like everything else this session on a single seed).

**Takeaway**: the factor that dominates every gap observed this session is
neither the algorithm (PPO vs REINFORCE) nor the new state feature -- it's
`lr`, which hadn't been correctly calibrated for either one before this
latest round of tests.

## `--no-counterfactual-baseline` ablation (PPO)

Under normal conditions the PPO critic regresses the COUNTERFACTUAL return
(`reward - counterfactual_reward(...)`, point 3 of `remarques_rl.md`), a
weak residual to predict (R2~0.06 measured earlier). On the RAW return
(`--no-counterfactual-baseline`), the critic has a much easier task
(R2~0.28 measured on this same raw return) -- tested to see whether a
richer critic signal translates into better final performance, same
hyperparameters otherwise (`lr=3e-5, epochs=8`, 50k episodes).

| | eval_avg(5000) | win_rate |
|---|---|---|
| ppo_50000 (with counterfactual) | +17.90 | 53.9% |
| ppo_50000 (`--no-counterfactual-baseline`) | +18.99 | 53.9% |

**No detectable difference** (delta +1.09, well under the ~+/-6 pt 95%
CI, same win_rate). Reading: the counterfactual and the critic both aim to
remove the same variance component ("donne luck"), just through different
routes -- the counterfactual via a privileged retrospective info (replaying
the whole donne with the heuristic) that no critic can reconstruct from a
partial state, the critic via a regression learned on the raw return. The
two methods seem to substitute for each other rather than stack, hence a
similar final result despite very different critic R2 between the two
configs.

Complementary diagnostic added on this occasion (measurement only, changes
no behavior): `clip_frac` in `train_ppo.py`'s logs, the fraction of samples
where the PPO clip (`|ratio-1| > clip_eps`) is actually active. Stays low
(~0.7-1.7%) throughout at `lr=3e-5` -- clipping rarely kicks in at this lr
regime.

## `--epochs 16` ablation (100k episodes): this session's best result

`clip_frac` staying low (~0.7-1.7%) at `epochs=8, lr=3e-5` suggested
clipping almost never kicked in -- so probably room to reuse each batch
even more before the clip actually starts holding things back. Tested
`epochs=16` (steps/episode goes from 2 to 4), otherwise same
hyperparameters, 100k episodes directly (no segments).

| | eval_avg(5000) | win_rate |
|---|---|---|
| **PPO epochs=16, 100k** | **+27.00** | **55.7%** |
| REINFORCE lr=3e-5, 100k | +25.14 | 55.3% |
| PPO epochs=8, 100k | +19.98 | 54.2% |

**This session's first delta to clearly exceed the 95% CI** (+7.02 vs
`epochs=8`, against ~+/-6 pts at n=5000) -- the most solid result obtained
so far, not just suggestive. `clip_frac` rises to ~2.5-4% (vs ~0.7-1.7%
at `epochs=8`), consistent with more batch reuse, but stays moderate -- no
entropy collapse (stable 0.15-0.22 throughout). PPO catches up to
REINFORCE `lr=3e-5` (even slightly above, delta +1.86, within the noise)
once this headroom is exploited -- confirming that clipping did still have
room to extract more learning per collected episode, PPO's own advantage
over REINFORCE.

Next to test if pushing further: `epochs=32` (watching whether `clip_frac`
keeps rising significantly, a sign we're approaching the point where the
clip starts really constraining things).

## `--no-critic-baseline` ablation (at `epochs=16`): the critic does matter, interacting with the epochs

Following the "the critic is useless, it's the batching that matters"
discussion (suggested by `--no-counterfactual-baseline` showing no
detectable effect, see above): a direct test removing the predicted-value
subtraction from the advantage (`--no-critic-baseline`, a new flag -- the
normalized advantage then becomes a centering on the current batch's mean,
REINFORCE-style but a per-batch scalar rather than a global EMA; `value_net`
is neither called nor trained), at `epochs=16, lr=3e-5` otherwise identical.

| | eval_avg(5000) | win_rate |
|---|---|---|
| PPO epochs=16, **with** critic | +27.00 | 55.7% |
| PPO epochs=16, **without** critic (`--no-critic-baseline`) | +18.89 | 54.1% |
| *for reference: PPO epochs=8, with critic* | *+19.98* | *54.2%* |

**The critic does matter** -- a delta of -8.11 when removed, above the
~+/-6 pt 95% CI (this session's 2nd delta to clearly exceed the noise,
after `epochs=16`'s own). This corrects/refines the previous conclusion:
it isn't "the critic is useless, only the batching matters" -- it's that
the critic and the epochs **interact**. Without the critic, `epochs=16`
(+18.89) falls back to `epochs=8`'s level **with** the critic (+19.98,
statistically indistinguishable): the gain previously attributed to "more
epochs" alone nearly evaporates without the critic.

**Interpretation**: the critic's role isn't to be accurate (R2~0.06 stays
weak, see above) but to slightly reduce the sample-by-sample noise of the
advantage. With more epochs, the SAME advantage estimates get reused
several times -- if they're noisy (centered on the batch's mean only),
each extra epoch amplifies that noise rather than extracting real signal;
with even an imperfect critic, this reuse becomes genuinely beneficial.
`--epochs`'s benefit therefore depends on the critic's presence, it isn't
an independent factor -- the opposite of what was thought after the
`--no-counterfactual-baseline` test (which only changed the critic's
TARGET, not its presence).

## Multi-seed validation: neither of the two conclusions above holds

The `epochs=16` and `--no-critic-baseline` ablations above each rested on
a single seed (20). Repeated with 2 more seeds (21, 22) for the 3 configs
(`epochs=8` critic, `epochs=16` critic, `epochs=16` without critic), same
eval protocol (`eval_policy.py`, 5000 games, seed=42):

| Config | seed20 | seed21 | seed22 | **mean** | **std** |
|---|---|---|---|---|---|
| epochs=8 (critic) | +19.98 | +23.45 | +20.34 | **21.26** | 1.91 |
| epochs=16 (critic) | +27.00 | +23.51 | +18.55 | **23.02** | 4.25 |
| epochs=16 (no critic) | +18.89 | +21.30 | +19.69 | **19.96** | 1.23 |

**Neither of the two previous conclusions survives**:
- `epochs=16` vs `epochs=8`: mean delta of +1.76, well within `epochs=16`'s
  own standard deviation (4.25) -- not significant.
- critic vs no critic (at `epochs=16`): mean delta of +3.06, same
  finding -- not significant.
- Worse: the ranking **flips** depending on the seed. At seed20,
  `epochs=16` critic dominates everything (+27.00). At seed22, it's the
  **worst** of the three (+18.55, even behind the no-critic version). The
  spectacular deltas measured earlier (+7.02 for `epochs=16`, +8.11 for the
  critic) rested on that exact seed20, a favorable draw for that
  combination -- not a systematic difference.

**Lesson**: seed-to-seed variance (standard deviation up to 4.25 pts) is
on the same order as the effects we thought we'd isolated on a single run.
The `epochs=16` and `--no-critic-baseline` sections above are kept as-is
(useful as an illustration of the risk), but their conclusions are
invalidated by this validation -- at this stage, these 3 configs can't be
reliably distinguished with only 3 seeds each. The only finding that
remains solid: all 3 configs (means 20-23) clearly beat both `heuristic`
(+8.07) and `imit.pt` (+1.91).

## Re-evaluation at 45000 games: much of the "seed noise" was measurement noise

The same 9 checkpoints, re-evaluated at `--games 45000` (instead of 5000)
to separate measurement noise (a finite game sample) from real training
noise (cf. the general discussion on evaluation precision).

**`heuristic` against itself goes from +8.07 (n=5000) to -0.51 (n=45000)**
-- much closer to what's expected by symmetry (0). The whole session
computed its deltas against a reference that was itself noisy at n=3000-5000.

| Config | seed20 | seed21 | seed22 | mean | std (was at n=5000) |
|---|---|---|---|---|---|
| epochs=8 (critic) | +14.81 | +15.56 | +15.01 | 15.13 | **0.39** (was 1.91) |
| epochs=16 (critic) | +14.07 | +15.42 | +10.84 | 13.44 | **2.35** (was 4.25) |
| epochs=16 (no critic) | +14.89 | +13.64 | +13.80 | 14.11 | **0.68** (was 1.23) |

**Confirmed**: for `epochs=8` and `no critic`, the standard deviation
collapses to nearly zero -- almost all of the variance measured at n=5000
was evaluation noise, not a real difference between seeds (the 3 seeds in
each group are now nearly indistinguishable).

**But `epochs=16` stays clearly more spread out** (2.35, vs 0.39 and
0.68) even at this precision -- this is no longer measurement noise, it's
a real signal: `epochs=16` appears to be an intrinsically **more unstable**
config from one seed to the next than `epochs=8`, consistent with the
intuition that reusing the batch more raises the risk of "sticking" to
that batch's specific noise rather than learning a signal that generalizes.

The 3 means (15.13 / 13.44 / 14.11) are now very close -- `epochs=16` is
even slightly the LOWEST of the three (the opposite of the initial "epochs=16
is the best" conclusion), even though that's not clear given its own
variance. **At this stage, none of the 3 configs clearly stands out from
the other two**; the only solid finding remains that all of them clearly
beat `heuristic` (now measured at -0.51, not +8.07).

## `epochs=16` at 7 seeds: the high variance is confirmed, not just a small-sample artifact

4 more seeds (23-26) for `epochs=16` only (`epochs=8` and `no critic` stay
at n=3 -- cf. the power-calculation discussion: that's enough for this
particular comparison, since the other two's variance is already very
low). Values at n=45000: 14.07, 15.42, 10.84, 8.29, 12.74, 16.97, 15.63.

**Mean ≈ 13.42, std ≈ 3.04** (n=7).

- **The mean has barely moved** (13.44 at n=3 -> 13.42 at n=7) -- it's now
  a fairly stable estimate.
- **The standard deviation hasn't dropped, it's even risen slightly**
  (2.35 at n=3 -> 3.04 at n=7) -- so this wasn't a small-sample artifact
  that would shrink with more seeds. `epochs=16` really does have a wide,
  now-confirmed spread (from +8.29 to +16.97 depending on the seed, a
  8.7-point gap).
- `epochs=16`'s mean (13.42) remains the **lowest** of the three configs,
  and the gap with `epochs=8` (1.71 pts, combined SE ~1.17, t≈1.46) is
  starting to become suggestive without being firm proof yet.

**Conclusion**: `epochs=16` isn't just "not proven better" as thought at
n=3 -- the additional data now lean more toward "probably not better, and
more unstable". `epochs=8` remains the most defensible default choice: a
mean at least as good, and a much lower standard deviation (0.39 vs 3.04)
-- a reliable result rather than a lottery between very good and mediocre.

## REINFORCE at 4 seeds per lr: "PPO beats REINFORCE" and "REINFORCE beats PPO" were both wrong

Same treatment as for `epochs`: 3 more seeds (21-23) for REINFORCE
`lr=3e-5` and `lr=1e-4` (in addition to the original seed), re-evaluated
at `--games 45000`.

| lr | values (n=4) | mean | std |
|---|---|---|---|
| `3e-5` | 15.39, 15.56, 14.67, 14.98 | **15.15** | **0.40** |
| `1e-4` | 14.34, 16.89, 13.87, 15.17 | **15.07** | 1.33 |

**The "lowering the lr also helps REINFORCE" conclusion collapses**: a
mean delta of 0.08 (vs +4.07 on the single-seed comparison, +25.14 vs
+21.07, which had looked clear). The two lrs actually give REINFORCE the
same performance.

**And this closes the whole v3 lineage's PPO vs REINFORCE comparison**:
`epochs=8` (PPO, validated over 3 seeds) gives a mean of **15.13**, std
**0.39**. REINFORCE `lr=3e-5` gives a mean of **15.15**, std **0.40**.
Nearly identical. Neither "PPO beats REINFORCE" (this lineage's initial
conclusion, one seed each) nor "REINFORCE beats PPO" (the following
conclusion, after fixing REINFORCE's lr, still on a single seed) held up
-- the two algorithms essentially converge to the same performance on this
task (~15 points above `heuristic`, now reliably measured at -0.51 rather
than +7 to +8 as at the start of the session).

## `--no-critic-baseline` re-evaluated at 45000: the "the critic matters" conclusion also collapses, and flips

The 3 `epochs=16` no-critic checkpoints (seed20/21/22, never re-evaluated
beyond n=5000) re-evaluated at `--games 45000` -- same protocol as the
rest of this section, no new training needed.

| | values (n=3) | mean | std |
|---|---|---|---|
| `epochs=16` with critic (n=7, see above) | 14.07, 15.42, 10.84, 8.29, 12.74, 16.97, 15.63 | 13.42 | 3.04 |
| `epochs=16` without critic (n=3) | 14.89, 13.64, 13.80 | **14.11** | **0.68** |

**The initial conclusion ("the critic matters", -8.11 delta on seed20 at
n=5000: +27.00 with critic vs +18.89 without) collapses and flips**:
`no critic` now has a slightly **higher** mean (14.11 vs 13.42, not
significant given the with-critic group's variance) -- the opposite of the
initial delta. The "+27.00" for seed20 was just an inflated measurement
draw (its real value, remeasured at n=45000, is 14.07). Bonus: `no critic`
keeps as low a standard deviation (0.68) as this session's other stable
configs (`epochs=8` PPO, REINFORCE `lr=3e-5`) -- it's `epochs=16` **with**
critic that's the unstable outlier, not removing the critic.

**At this stage, as with `--no-counterfactual-baseline`, no evidence that
the critic brings anything measurable** to this task -- this session's
three attempts to make it useful (pretraining, a richer target,
presence/absence) never held up once tested rigorously.

## Centralized critic (CTDE): `encode_full_state` (263-dim) reserved for ValueNet

The partial-information critic plateaus (R2~0.04-0.06, cf. the previous
section) because of hidden information (opponents' hands). So a
**centralized** critic was tested: `ValueNet` now takes `encode_full_state`
(`coinche/rl_agent.py`) = `encode_state` (167-dim, what the actor sees) +
the CURRENT hands of the 3 other seats (3*32=96-dim) = 263-dim
(`FULL_STATE_DIM`). Only the critic sees this privileged information -- the
actor (`CardNet`, an independent network) keeps deciding from `encode_state`'s
167 dimensions alone, exactly as before. Available only because training
simulates the whole donne (all 4 hands are known to the process); never
usable in real play. Analogous to a human player who reviews their donne
afterward with all the information revealed to better judge their choices,
without that changing what they could see at the moment of playing.

Same protocol as the rest of this section: 3 seeds (20/21/22), same
hyperparameters as the already-validated `epochs=8` config (`lr=3e-5,
epochs=8, 100k episodes`), evaluated on the same 45000 fixed donnes
(`seed=42`).

| | values (n=3) | mean | std |
|---|---|---|---|
| partial critic (`encode_state`, epochs=8, reference) | 14.81, 15.56, 15.01 | 15.13 | 0.39 |
| centralized critic (`encode_full_state`, same hyperparameters) | 16.25, 16.32, 17.69 | **16.75** | 0.81 |

**Mean delta of +1.63** (~11% better than the +15.13 advantage over
`heuristic`), and **complete separation** between the two groups: the
centralized group's minimum (16.25) exceeds the partial group's maximum
(15.56) -- the clearest signal an n=3 vs n=3 design can produce (a
one-sided permutation test on ranks: p=0.05 exactly, the best value
attainable at this sample size). A Welch's t-test on these same figures
gives t=3.13 (df≈2.9), just BELOW the conventional two-tailed threshold
(t_crit≈3.18 at df=3) -- borderline, in the same vein as the rest of this
section.

A point of caution: the centralized group's standard deviation (0.81) is
already roughly double that of this lineage's stable configs (0.39-0.40
for `epochs=8` PPO and REINFORCE `lr=3e-5`) -- the same warning sign that,
for `epochs=16`, turned out to foreshadow real seed instability once
measured at n=7 (std 3.04). Nothing says that's the case here, but nothing
rules it out at n=3 either.

**Cautious verdict**: the centralized critic seems to help (+1.63 on
average, complete separation of the 3+3 seed values) -- this is, at this
point in the session, the only critic attempt showing an effect in the
expected direction rather than collapsing under rigorous testing. But
n=3 vs n=3 is the smallest justifiable base in the methodology applied
throughout this README: it isn't yet at the confidence level of the other
settled conclusions here (`epochs=8` vs REINFORCE, `epochs=16` less
stable, the partial critic showing no effect). 3-4 more seeds on each side
would confirm -- or deflate, like so many other promising results this
session -- this delta.

## Centralized critic re-tested at n=7: the delta deflates and becomes non-significant again

4 more seeds (23/24/25/26, exactly the same protocol), bringing the
centralized-critic group to n=7 -- the same approach that had revealed
`epochs=16`'s real instability (variance underestimated at n=3).

| | values | mean | std |
|---|---|---|---|
| centralized critic (n=3, seeds 20-22) | 16.25, 16.32, 17.69 | 16.75 | 0.81 |
| centralized critic (n=7, seeds 20-26) | 16.25, 16.32, 17.69, 15.89, 17.32, 13.02, 15.34 | **15.98** | **1.53** |
| partial critic (n=3, reference) | 14.81, 15.56, 15.01 | 15.13 | 0.39 |

**The delta deflates and is no longer significant**: a mean of 15.98
against 15.13 for the partial critic (delta +0.85, vs +1.63 measured at
n=3), and the centralized group's standard deviation has nearly doubled
(1.53 vs 0.81) -- seed25 alone (13.02, the lowest of the 7) would have
been enough on its own to break the complete separation seen at n=3. A
Welch's t-test on these figures gives t=1.37 (df≈7.4), far from the
threshold (t_crit≈2.36 at df=7): the difference is no longer
distinguishable from noise.

**The same lesson as for `epochs=16`**: a standard deviation that looks
low at n=3 can just be lucky sampling, not a stable property of the
config -- the standard deviation can't be reliably measured with so few
seeds. **Final verdict (pending a larger sample): the centralized critic
shows no measurable advantage** over the partial critic. This is the
session's fourth attempt to make the critic useful (after supervised
pretraining, an enriched counterfactual target, a presence/absence test),
and the fourth time the initial signal doesn't survive a larger sample.

## Ablation of the "trick points banked per side" feature (`_points_so_far`)

This feature (2 scalars in `encode_state`, added on the intuition that a
human player accounts for it, see above) had never been isolated from the
other changes that happened at the same time (lr, state dimension). Tested
properly here: same protocol everywhere (`epochs=8, lr=3e-5,
--no-critic-baseline` -- chosen because the critic, in every form tested
in this README, shows no measurable effect, so might as well remove this
extra confound), 3 seeds (20/21/22) per arm, `imit.pt`/`imit_ablated.pt`
regenerated identically (same pretrain hyperparameters, seed 0) so neither
arm ever "saw" the feature differently from the other from the start.
`ablate_points=True` forces the `_points_so_far` block to zero (actor AND
critic if not disabled) without changing `STATE_DIM`, cf.
`coinche/rl_agent.py`/`train_ppo.py --ablate-points-so-far`.

| | values (n=3) | mean | std |
|---|---|---|---|
| with the feature | 17.47, 16.26, 13.44 | 15.72 | 2.07 |
| without the feature (ablated) | 14.06, 15.69, 15.61 | 15.12 | 0.92 |

**No measurable effect**: a mean delta of +0.60, well within both groups'
noise (standard deviations 2.07 and 0.92 -- much wider than the delta
itself). A Welch's t-test gives t=0.46, far from any significance
threshold. The feature makes the actor neither gain nor lose, once
isolated from every confound -- consistent with the already-established
conclusion on the critic (no form of critic tested in this README ever
showed a measurable effect, and this feature had been added precisely on
the idea it would help the critic). It stays in `encode_state` by default
(the intuition that motivated it remains valid even without proof of a
measured gain), but it doesn't explain any performance difference observed
elsewhere in this README.

## Growing pool + PFSP, retested on the current state/scoring (`run_growing_pool.py`)

Already tried once in the v2 lineage (the old scoring, before the rule-5
fix, single seed, evaluated only at n=3000): `exp_growing_pool` there
showed a peak at segment 6/10 (300k ep.) followed by a non-significant
decline -- a lukewarm conclusion, never reconfirmed on the current
state/scoring nor with the rigor (n=45000, multi-seed) established earlier
in this README.

Retested here: seed20, 300k episodes, 6 segments of 50k
(`run_growing_pool.py --total-episodes 300000 --segment-episodes 50000
--init-load rl_experiments_v3/imit.pt --seed-start 20`), settings
unchanged (`lr=1e-4, entropy-beta=0.002`, `--pfsp-refresh-every 1000
--pfsp-temperature 0.1 --pfsp-ema-beta 0.98`, pool = heuristic + every
previous segment's checkpoint). Real duration: 5603s (~1h33) for the 6
segments (670, 892, 970, 1020, 1014, 1037s -- growing with the pool size,
the same behavior as v2).

Evaluated at n=45000 against heuristic alone (same protocol as the rest of
this README):

| Checkpoint | eval_avg(45000) | win_rate |
|---|---|---|
| imit.pt | -2.10 | 49.5% |
| seg_50000 | +12.32 | 52.4% |
| seg_100000 | +15.74 | 53.1% |
| **seg_150000** | **+17.81** | **53.6%** |
| seg_200000 | +15.18 | 53.0% |
| seg_250000 | +13.87 | 52.9% |
| seg_300000 | +12.46 | 52.6% |

The same pattern as in v2: it rises, peaks at the midpoint (seg_150000,
50% of the run), then declines -- here dropping back down to almost
`seg_50000`'s level. The final checkpoint is **below** the stable plateau
obtained by plain training against heuristic alone, with no pool
(`epochs=8` PPO or REINFORCE, ~15.1-15.2 over 3-4 seeds, see above).

**But `eval_avg` against heuristic alone only measures the exploitation of
a fixed opponent, not versatility** -- exactly what a growing pool is
supposed to trade off a little for the sake of robustness. Checked via a
policy-vs-policy round-robin (`eval_matchup.py`, a new script: the same
`generate_fixed_deals`/`evaluate_fixed` as `eval_policy.py`, but pitting
two checkpoints against each other instead of a checkpoint against
heuristic), n=10000 donnes per matchup, between `heuristic`, the "vanilla"
checkpoint (`exp_ppo_lr3e-5/ppo_100000.pt`, trained only against
heuristic), `seg_50000`, `seg_150000` (the peak vs heuristic), and
`seg_300000` (the final one):

| A \ B (seat 0/2 vs 1/3) | heuristic | vanilla | seg50k | seg150k | seg300k |
|---|---|---|---|---|---|
| heuristic | -- | -11.46 | -9.42 | -13.01 | -9.29 |
| vanilla | +17.63 | -- | +6.87 | -0.24 | -1.32 |
| seg50k | +14.89 | -1.70 | -- | -4.97 | -5.89 |
| seg150k | +22.46 | +5.88 | +12.54 | -- | +0.95 |
| seg300k | +15.77 | +3.51 | +10.29 | +4.56 | -- |

**The hypothesis is confirmed**: `seg300k` beats `vanilla` both ways
(+3.51 / -1.32) even though `vanilla` scores better against heuristic
alone in this same pass (+17.63 vs +15.77) -- the network specialized in
exploiting heuristic loses head-to-head against a less specialized but
more versatile network. And `seg300k` is nearly tied with `seg150k`
head-to-head (+0.95 / +4.56) even though `seg150k` scores much better
against heuristic alone (+17.81 vs +12.46) -- so the vs-heuristic table's
"decline" does **not** correspond to a drop in general skill, only a shift
in specialization (less pure exploitation of heuristic, without losing
strength against other styles). `seg300k` also clearly beats `seg50k`
both ways (+10.29 / -5.89): the pool's progress is real, not noise.

Caveat: a single seed (as always in this README for a first pass), and
the two directions of a given round-robin matchup aren't perfectly
symmetric (e.g. vanilla vs seg50k: +6.87 then -1.70) -- seat-placement
noise on the same donnes, to be read as a direction rather than an exact
value.

**To redo**: 2 more seeds (same config, seed21/22) to confirm this pattern
(real progress in versatility, despite a misleading decline in `eval_avg`
vs heuristic alone) reproduces -- and if confirmed, add a larger
round-robin (more segments, + the REINFORCE checkpoint
`exp_reinforce_lr3e-5`) to fully characterize the versatility gained.
Deferred to a later session (~1h33/seed, ~7h for 3 full 300k seeds).

### Seed21: the pattern is confirmed, on a completely different `eval_avg` trajectory

Seed21 (exactly the same config, `--seed-start 21`): 5519s (~1h32).
`eval_avg` vs heuristic alone this time draws a near-continuous rise (no
peak-then-collapse like seed20):

| Checkpoint | eval_avg(45000) | win_rate |
|---|---|---|
| seg_50000 | +13.60 | 52.7% |
| seg_100000 | +15.74 | 53.2% |
| seg_150000 | +16.90 | 53.4% |
| seg_200000 | +19.37 | 53.9% |
| seg_250000 | +22.25 | 54.4% |
| seg_300000 | +20.81 | 54.2% |

Two seeds, two very different shapes of `eval_avg` vs heuristic curve
(peak-collapse for seed20, rise-slight-pullback for seed21) -- further
confirmation that this single metric is too noisy/too narrow (a single
fixed opponent) to judge pool training.

Extended round-robin (`eval_matchup.py`, n=10000/matchup): heuristic,
vanilla, seed20/seg_50000, seed20/seg_300000, seed21/seg_50000,
seed21/seg_300000 (6 specs, 30 matchups):

| A \ B | heuristic | vanilla | seed20/50k | seed20/300k | seed21/50k | seed21/300k |
|---|---|---|---|---|---|---|
| heuristic | -- | -11.46 | -9.42 | -9.29 | -8.59 | -16.17 |
| vanilla | +17.63 | -- | +6.87 | -1.32 | +5.74 | -5.05 |
| seed20/50k | +14.89 | -1.70 | -- | -5.89 | +0.44 | -11.84 |
| seed20/300k | +15.77 | +3.51 | +10.29 | -- | +6.37 | +0.31 |
| seed21/50k | +17.91 | +1.59 | +4.35 | -1.01 | -- | -7.21 |
| seed21/300k | +23.80 | +9.86 | +16.92 | +4.19 | +13.03 | -- |

**A methodological discovery along the way**: the sum of a given matchup's
two directions (X vs Y + Y vs X) is almost never zero -- there's a
systematic seat-placement bias of about +2 to +5 points favoring whoever
occupies seats 0/2 (probably tied to the dealer rotation in
`generate_fixed_deals`), independent of the two players' real skill. Since
everything else in this README always compares RL at 0/2 against heuristic
at 1/3, this bias never skewed any earlier comparison (applied identically
to every checkpoint) -- but in a round-robin where each checkpoint takes
turns at both seats, it needs to be neutralized by averaging both
directions of each pair: `skill(X,Y) = ((X vs Y) - (Y vs X)) / 2`.

Each one's average strength against the other 5, once this bias is
corrected:

| | corrected average strength |
|---|---|
| **seed21/seg_300000** | **+10.78** |
| seed20/seg_300000 | +4.96 |
| vanilla | +2.21 |
| seed21/seg_50000 | -0.14 |
| seed20/seg_50000 | -3.31 |
| heuristic | -14.49 |

**The pattern is confirmed across the 2 seeds**: both final (300k)
growing-pool checkpoints beat vanilla in real strength, despite completely
different `eval_avg`-vs-heuristic trajectories on the surface (collapse
for seed20, near-plateau for seed21). Both early (50k) checkpoints stay
close to each other and below vanilla. There's real seed variance on the
final level reached (seed21 clearly above seed20), but the qualitative
conclusion -- the growing pool produces a network that's generally
stronger than plain training, even when `eval_avg` vs heuristic alone
suggests the opposite -- now holds across 2 independent seeds.

**To redo**: seed22 (the last of the 3 planned) for confirmation at n=3;
consider extending the round-robin to plain REINFORCE
(`exp_reinforce_lr3e-5`) and to PPO in a growing pool
(`run_growing_pool.py` currently only supports `train.py`/REINFORCE --
never tested with `train_ppo.py`).

### Seed22 (n=3): final conclusion -- confirmed without exception across the 3 seeds

Seed22 (same config, ~1h28): `eval_avg` vs heuristic alone draws a third,
still different curve shape -- a rise then a **stable plateau** around +20
(150k to 300k), with neither a collapse (seed20) nor a pullback (seed21):

| Checkpoint | eval_avg(45000) | win_rate |
|---|---|---|
| seg_50000 | +11.85 | 52.3% |
| seg_100000 | +16.54 | 53.3% |
| seg_150000 | +20.05 | 54.0% |
| seg_200000 | +20.40 | 54.1% |
| seg_250000 | +20.36 | 54.1% |
| seg_300000 | +20.24 | 54.1% |

Three seeds, three qualitatively different `eval_avg` trajectories
(collapse / pullback / plateau) -- further confirmation that this metric
alone isn't enough to judge pool training.

**Checking the seat bias (in passing)**: before extending the round-robin,
checked that the +2 to +5 point "bias" favoring seats 0/2 (noted in the
seed21 section) isn't a real mechanical bias but an artifact of this
specific 10000-donne sample. Rotated the whole table by a quarter turn
(hands AND dealer shifted together, same donnes): the results flip
**exactly** (`rotated(A vs B) = -original(B vs A)`, verified to the
decimal on several pairs) -- proof the engine treats the 4 seats perfectly
symmetrically, and the observed "bias" is specific to this particular
donne sample (seed=42), not a property of the game. The already-applied
correction `skill(X,Y) = ((X vs Y) - (Y vs X))/2` is therefore the right
method.

Round-robin extended to 8 specs (heuristic, vanilla, seed20/21/22 x
early/final, n=10000/matchup -- only the 26 new pairs involving seed22
were replayed, the rest reuses previously obtained results):

| A \ B | heuristic | vanilla | s20-50k | s20-300k | s21-50k | s21-300k | s22-50k | s22-300k |
|---|---|---|---|---|---|---|---|---|
| heuristic | -- | -11.46 | -9.42 | -9.29 | -8.59 | -16.17 | -7.54 | -17.65 |
| vanilla | +17.63 | -- | +6.87 | -1.32 | +5.74 | -5.05 | +6.03 | -8.88 |
| s20-50k | +14.89 | -1.70 | -- | -5.89 | +0.44 | -11.84 | +2.24 | -14.24 |
| s20-300k | +15.77 | +3.51 | +10.29 | -- | +6.37 | +0.31 | +9.51 | -3.69 |
| s21-50k | +17.91 | +1.59 | +4.35 | -1.01 | -- | -7.21 | +2.48 | -10.43 |
| s21-300k | +23.80 | +9.86 | +16.92 | +4.19 | +13.03 | -- | +13.42 | -2.07 |
| s22-50k | +15.03 | +0.93 | +4.35 | -3.17 | +6.25 | -6.52 | -- | -9.87 |
| s22-300k | +22.15 | +9.56 | +13.48 | +6.40 | +10.26 | +5.04 | +12.58 | -- |

Final ranking (seat-bias-corrected average strength, against the other 7):

| | average strength |
|---|---|
| **seed22/300k** | **+10.45** |
| seed21/300k | +8.61 |
| seed20/300k | +3.73 |
| vanilla | +0.62 |
| seed22/50k | -1.54 |
| seed21/50k | -1.84 |
| seed20/50k | -4.50 |
| heuristic | -14.81 |

**Final conclusion (n=3, the same rigor as the rest of this README):** all
3 final (300k) checkpoints are above vanilla, and all 3 early (50k)
checkpoints are below -- without exception. The growing pool + PFSP
(`run_growing_pool.py`, REINFORCE) produces a network that's measurably
more versatile/generally stronger than plain training against heuristic
alone -- even though `eval_avg` against heuristic alone, taken in
isolation, can suggest the opposite depending on the seed (cf. seed20).
This is the first structural idea in the whole v3 lineage whose initial
effect didn't deflate under full sampling -- on the contrary, it got
confirmed and clarified.

**To redo if pushing further**: extend to plain REINFORCE
(`exp_reinforce_lr3e-5`, never included in the round-robin) and build a
PPO equivalent of the growing pool (`run_growing_pool.py` currently only
drives `train.py`).

### Plain REINFORCE added to the round-robin: the ranking still holds

Added `exp_reinforce_lr3e-5/reinforce_100000.pt` (never included until
now) to the round-robin -- 16 new matchups (against the 8 already-tested
specs, n=10000, same donnes), reusing the already-known pairs for the rest.

Corrected average strength (9 specs, against the other 8):

| | average strength |
|---|---|
| **seed22/300k** | **+10.03** |
| seed21/300k | +8.46 |
| seed20/300k | +3.62 |
| plain REINFORCE | +1.05 |
| vanilla (plain PPO) | +0.48 |
| seed22/50k | -1.53 |
| seed21/50k | -1.95 |
| seed20/50k | -4.42 |
| heuristic | -15.10 |

Plain REINFORCE ranks just above vanilla PPO (+1.05 vs +0.48) --
consistent with the `PPO≈REINFORCE` already established earlier in this
README -- and above all stays, like vanilla, clearly below the 3 final
growing-pool checkpoints and above the 3 early ones. So the conclusion
doesn't depend on which algorithm is chosen as the "plain training"
reference: plain PPO and plain REINFORCE both rank at the same
intermediate level, far behind the growing pool's final checkpoints.

## Growing pool in PPO (`run_growing_pool_ppo.py`): the gain shows up there too

`run_growing_pool.py` only drove `train.py`/REINFORCE. Wrote
`run_growing_pool_ppo.py` (same segment logic, same pool + PFSP) but
targeting `train_ppo.py` with the already-validated stable PPO config
(`epochs=8, lr=3e-5, value-coef=0.5, entropy-coef=0.002`) -- to check
whether the gain confirmed on REINFORCE (n=3) also holds with PPO.

**Bug found and fixed before launching anything**: the opponent pool
(`--opponent`) loads every checkpoint via `NeuralPolicy.load()`
(`coinche/rl_agent.py`), which expected a raw CardNet state_dict -- not the
native PPO format (`{'policy':, 'value':}`, cf. `PPOPolicy.save()`). From
segment 2 onward, the PPO orchestrator would have crashed trying to load
the previous segment's checkpoint as a frozen opponent. Fixed:
`NeuralPolicy.load()` now accepts both formats (`obj['policy'] if 'policy'
in obj else obj`), the same logic as `PPOPolicy.load()`. Smoke-tested (400
episodes, 2 segments) before launching the real run.

Segment continuation: `--load` AND `--load-value` both receive the
previous segment's checkpoint (policy + critic), except for the very
first segment (starts from `imit.pt`, policy only, critic starts random).

First seed (20, 300k episodes, 6 segments, ~1h46): `eval_avg` vs heuristic
alone rises cleanly (+13.29 -> +13.98 -> +17.53 -> +19.10 -> +18.66 ->
**+19.86**), with no collapse -- a shape similar to REINFORCE's cleanest
seeds (21/22).

Round-robin extended to 11 specs (heuristic, vanilla PPO, plain
REINFORCE, pool REINFORCE seed20/21/22 x early/final, pool PPO seed20 x
early/final; only the 38 new pairs involving the PPO pool were replayed,
n=10000, same donnes as the rest of this section):

| | average strength |
|---|---|
| **pool PPO/seed20 final** | **+9.49** |
| pool REINFORCE/seed22 final | +8.97 |
| pool REINFORCE/seed21 final | +7.44 |
| pool REINFORCE/seed20 final | +2.95 |
| plain REINFORCE | +0.35 |
| vanilla (plain PPO) | -0.41 |
| pool REINFORCE/seed22 early | -2.12 |
| pool REINFORCE/seed21 early | -2.68 |
| pool PPO/seed20 early | -3.37 |
| pool REINFORCE/seed20 early | -4.71 |
| heuristic | -15.40 |

**Confirmed without exception, regardless of algorithm**: the 4 final
(300k, 1 PPO + 3 REINFORCE) checkpoints are all above both plain training
runs (plain REINFORCE and plain PPO), and the 4 early (50k) checkpoints
are all below. The first PPO seed even ranks first of the whole ranking --
on a single PPO seed, not to be overinterpreted beyond "PPO also clearly
benefits from the growing pool", but the qualitative pattern (growing pool
> plain training) doesn't depend on which plain-training algorithm is
used as the reference, nor on which algorithm drives the pool itself.

**To redo to also confirm at n=3 for PPO**: 2 more PPO seeds
(`run_growing_pool_ppo.py --seed-start 21` then `22`), the same approach
as for REINFORCE.

### Seed21 PPO (n=2): the same exceptionless pattern

Seed21 PPO (same config, ~1h40): `eval_avg` vs heuristic alone rises with
a slight zigzag (+13.25 -> +16.50 -> +16.47 -> +19.32 -> +17.26 ->
**+18.36**), consistent with seed20 (+19.86).

Round-robin extended to 13 specs (heuristic, vanilla PPO, plain
REINFORCE, pool REINFORCE seed20/21/22 x early/final, pool PPO seed20/21
x early/final; only the 46 new pairs involving seed21 PPO were replayed,
n=10000, same donnes as the rest of this section). Seat-bias-corrected
average strength, against the other 12:

| | average strength |
|---|---|
| **pool PPO/seed20 final** | **+8.76** |
| pool PPO/seed21 final | +8.40 |
| pool REINFORCE/seed22 final | +8.37 |
| pool REINFORCE/seed21 final | +6.95 |
| pool REINFORCE/seed20 final | +2.28 |
| plain REINFORCE | -0.28 |
| vanilla (plain PPO) | -1.05 |
| pool PPO/seed21 early | -2.37 |
| pool REINFORCE/seed22 early | -2.60 |
| pool REINFORCE/seed21 early | -3.17 |
| pool PPO/seed20 early | -3.98 |
| pool REINFORCE/seed20 early | -5.23 |
| heuristic | -15.61 |

**Without exception, across 13 entities**: the 5 final checkpoints (2 PPO
+ 3 REINFORCE) occupy the top 5 spots, far ahead of both plain-training
runs; the 5 early checkpoints are all below both plain runs. The top 3
(PPO seed20/21, REINFORCE seed22 -- +8.76/+8.40/+8.37) are essentially
tied with each other, but all clearly separated from the plain group.
Now confirmed at n=2 for PPO with the same exceptionless pattern as
REINFORCE at n=3.

**To redo**: seed22 PPO (the last of the 3 planned) for a full n=3,
symmetric with REINFORCE.

## To redo in this lineage if continuing

- If really wanting to distinguish `epochs=8` (PPO) from REINFORCE
  `lr=1e-4` (the only group still with a notable standard deviation,
  1.33) from the two minimum-variance options (`epochs=8` PPO, REINFORCE
  `lr=3e-5`), more seeds would be needed -- but the essential point is
  already settled: neither algorithm dominates the other on this task.
- ~~Confirm (or deflate) the centralized critic's delta~~: done, 4 more
  seeds (n=7 total) -- the delta deflates (+0.85, not significant), the
  same conclusion as the 3 other critic attempts. Only remains open if
  also wanting to bring the partial-critic group to n=7 for a fully
  symmetric comparison -- little reason to expect a change in conclusion.
