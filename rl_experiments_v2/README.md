# RL checkpoints - lineage v2 (post rule 5)

A directory separate from `rl_experiments/`, created when `GameEngine.play()`'s
scoring (coinche/game.py) was fixed to implement the full rule 5 from
`regles_coinche.md`: falling short (attack 0 / defense 160+contract,
instead of simply returning the raw trick points without accounting for
contract success), rounding to the nearest ten, the 500-point capot bonus,
the TA scale (x1.5).

Every checkpoint in `rl_experiments/` (`imit.pt` through `exp_pool_350k`)
was trained against the **old** signal, which didn't specifically penalize
falling short beyond the natural card-point deficit -- they remain valid as
a record of that first lineage (see its README), but are no longer the
reference for new training runs.

`imit.pt` (supervised imitation of `HeuristicPlayer`, independent of the
reward and so still valid) remains the common starting point. Every new RL
experiment (self-play, pool, PFSP...) restarts from here, on the corrected
signal.

## `exp_growing_pool/`: growing pool + PFSP (`run_growing_pool.py`)

Segment-based orchestration (see `run_growing_pool.py` at the repo root):
each segment adds its own final checkpoint to the next segment's opponent
pool, with `train.py --pfsp` biasing sampling toward the currently
toughest pool opponent (the trainee's lowest win rate against them) rather
than uniform sampling.

Settings used (per the remarques_rl.md discussion):
- 50k episodes per segment
- `--pfsp-refresh-every 1000`, `--pfsp-temperature 0.1`, `--pfsp-ema-beta 0.98`
- `lr=1e-4`, `entropy-beta=0.002`, `entropy-decay=none` (factors already
  validated in the v1 lineage -- see the lr/entropy ablation in
  `rl_experiments/README.md`)

The detail (number of segments run, results) will be documented here as it
progresses.

### Results: full 500k run (10 segments of 50k, 2026-07-20)

Evaluation with `eval_policy.py` (fixed donnes, paired comparison, 3000
games, seed=42) of `heuristic`, `imit.pt`, and every segment checkpoint
against `HeuristicPlayer`:

| Checkpoint | avg vs heuristic | win_rate | delta vs imit.pt |
|---|---|---|---|
| heuristic (reference) | +2.4 +/- 8.3 | 50.7% | -- |
| imit.pt | +0.2 +/- 8.2 | 50.3% | -- |
| seg_50000 | +14.6 | 53.1% | +14.4 (signif.) |
| seg_100000 | +14.2 | 53.3% | +14.0 (signif.) |
| seg_150000 | +16.4 | 53.7% | +16.2 (signif.) |
| seg_200000 | +21.1 | 54.7% | +21.0 (signif.) |
| seg_250000 | +24.4 | 55.1% | +24.2 (signif.) |
| **seg_300000** | **+26.0** | **55.7%** | **+25.8 (peak)** |
| seg_350000 | +18.1 | 54.0% | +17.9 (signif. drop vs seg_300000: -7.95) |
| seg_400000 | +22.6 | 54.9% | +22.4 |
| seg_450000 | +23.7 | 55.1% | +23.5 |
| seg_500000 (final) | +23.5 | 55.0% | +23.3 |

95% CI ~ +/-7 to 8 points on these deltas (empirical std ~228 points/donne,
n=3000 -- much higher than the ~124 of the pre-rule-5 v1 lineage: rule 5
introduces much more violent swings, the 500 capot, doubled coinche,
asymmetric falling short).

**Findings:**
- `imit.pt` alone doesn't beat `heuristic` (expected, it's a clone of the
  heuristic).
- **Segment 1** (50k episodes, vs heuristic only, a 1-member pool) captures
  most of the gain on its own (+14.4, highly significant) -- the clearest
  jump of the whole experiment.
- From segment 2 onward (growing pool + PFSP active), segment-to-segment
  progress is almost entirely **not significant** (+/-6 to 8 points of
  noise): no reliable, steady progress once the pool kicks in.
- A peak at `seg_300000` (episode 300k), then a significant drop at
  `seg_350000` (-7.95, the only truly significant negative
  segment-to-segment delta) -- real instability, not just evaluation noise.
- The final checkpoint `seg_500000` is significantly better than
  `seg_50000` (+8.8) and than `imit.pt` (+23.3) -- so there is net progress
  over the whole run.
- But `seg_500000` is **not** significantly better than `seg_200000` (+2.3,
  not signif.) and is even slightly (not significantly) below the
  `seg_300000` peak (-2.6).

**Interpretation:** nearly all of the net exploitable gain happened within
the first ~200-300k episodes; the last 200-300k (segments 5 to 10, the pool
growing from 5 to 11 members) produced no statistically detectable
improvement, with even a clear hiccup around 350k. This is better than
`exp_selfplay_frozen150k`/`exp_pool_350k`'s catastrophic plateaus (no
collapse), but the "growing pool + PFSP avoids plateauing" hypothesis isn't
clearly confirmed by these numbers beyond ~300k.

## PPO (`train_ppo.py`): comparison against REINFORCE

`train_ppo.py` implements PPO (Schulman et al. 2017) on exactly the same
task as `train.py`/REINFORCE (same state/action, same RL team at seats
0/2, same point-3 counterfactual from `remarques_rl.md`), to compare the
two algorithms under equal protocol. Round 1: `--load imit.pt --opponent
heuristic` alone (no pool/PFSP), 50k episodes -- the scale of REINFORCE's
segment 1 above (+14.4 vs `imit.pt`).

### Two bugs found and fixed before any usable result

The very first run (`exp_ppo_round1_buggy_valueloss`) drove the policy
below zero (eval_avg negative throughout, entropy climbing without
converging, 0.56 to 1.03). Two independent causes:

1. **`value_loss`'s scale**: MSE on raw returns (~hundreds of points)
   dominated `policy_loss` (on the normalized advantage, O(1)) by several
   orders of magnitude in the total loss -- the gradient flowing through
   the shared trunk ended up almost only minimizing the value error. Fix:
   divide `values`/`returns` by the batch's standard deviation before the MSE.
2. **Legal-move mask not reapplied** (`exp_ppo_round1_buggy_mask`, after
   fix #1 -- still degraded, even worse): `choose_card` samples under a
   masked distribution (illegal moves at `-inf`), but `update_batch`
   recomputed `new_log_probs` under the **unmasked** logits -- the PPO
   ratio (`exp(new_log_prob - old_log_prob)`) no longer meant "how much has
   the policy moved". Fix: store the mask in the trajectory, reapply it on
   recomputation. Verified with a direct test: ratio within [0.999999,
   1.000002] right after collection (before any gradient step), as
   expected if the mask is consistent on both sides.

Round 1 redone with both fixes (`exp_ppo_round1_epochs4`): finally healthy
(entropy stable ~0.2, `value_loss` stable ~1.0 normalized), eval_avg(3000)
via `eval_policy.py` = **+7.84** against **+15.22** for REINFORCE
(`seg_50000`) at the same budget -- clear progress over `imit.pt` (-3.08)
but barely above `heuristic`'s structural noise against itself (+7.38).

### `--epochs` ablation (batch reuse)

With `epochs=4, minibatch=64`, the number of gradient steps per episode
(`epochs x 16 / minibatch`, 16 = timesteps recorded per donne, 2 RL seats
x 8 tricks) comes out to 1.0 -- the same as REINFORCE, which underuses
PPO's own advantage (reusing a batch several times thanks to clipping).
`epochs=8` (2 steps/episode): eval_avg=**+9.75**
(`exp_ppo_round1_epochs8`) -- better, but the gain (~2 pts) stays under
the empirical 95% CI (~+/-7-8 pts, n=3000): not significant on a single run.

### Idea explored and dropped (for now): critic pretraining

Diagnosis (see also `remarques_rl.md`): the shared-trunk critic
(`epochs8`) only explained R2=0.043 of the raw return's variance, and
barely separated attack/defense (`is_attacker`, even though it's a direct
input feature): ~16 points captured out of a real gap of ~197. Likely
cause: its trunk, inherited from `imit.pt`, had never seen a value signal
before PPO, competing with `policy_loss` on that same shared trunk.

Architectural fix: `ActorCriticNet` (shared trunk) replaced by `CardNet`
(policy, = `imit.pt`, no remapping needed to load) + `ValueNet` (an
**independent** trunk), no more gradient competition. `pretrain_value.py`
(new, modeled on `pretrain.py`) pretrains `ValueNet` by supervised MSE
regression.

- **v1 (buggy target)**: regresses on the **raw** return
  (`team_points[0]-team_points[1]`, donnes from `HeuristicPlayer` x4) --
  good offline R2 (0.33, 5000 donnes/20 epochs), but PPO's `value_loss`
  actually applies to `reward - counterfactual_reward(...)` (already
  applied in the main loop before the critic comes in, cf. point 3).
  Verified empirically: the attack/defense gap drops from ~190 points on
  the raw return to only ~17 points on the counterfactual return -- the
  latter already absorbs most of the easy signal. So the v1 critic started
  out with confident predictions (std(value)=127) but **systematically
  biased** for the true PPO target. PPO round result
  (`exp_ppo_round1_valuepretrain`): eval_avg=**+4.92**, worse than
  epochs4/epochs8.
- **v2/v3 (fixed target)**: regresses on the counterfactual return,
  collected with the policy at seats 0/2 (like `run_episode`) rather than
  `HeuristicPlayer` x4. Much weaker residual signal (R2~0.05 at 5000
  donnes/20 epochs) -- confirmed real but data-starved by a train/val test
  at 20000 donnes/60 epochs (R2(val) climbs cleanly from 0.008 to 0.063,
  not noise). Final `value_imit.pt` (on disk): 20000 donnes, 60 epochs,
  unbiased group averages but a noisy R2 measurement on a small fresh
  sample (~-0.05 over 800 independent validation donnes -- expected given
  how weak the signal is). PPO round result
  (`exp_ppo_round1_valuepretrain_v3`): eval_avg=**-0.34** -- worse than the
  randomly initialized critic (`epochs8`, +9.75).

**Conclusion (idea on hold):** even correctly targeted and unbiased, a
critic with such a weak signal (R2~0.06) can do more harm than good -- its
own errors (std(value) measured between 32 and 57) potentially add more
noise to the advantage than a near-null critic (random init, close to "no
baseline"). But a single run per configuration (50k episodes, one seed)
can't settle whether it's "a real but harmful signal" or "run noise" (this
session's runs range from -0.34 to +15.22 for similar configs, on the same
order as the empirical 95% CI) -- would need several seeds to conclude
firmly. The architecture (independent `ValueNet`, `pretrain_value.py
--load-value`) and checkpoints are kept in case this idea is picked back
up later.

### `--lr` ablation

Starting from `epochs8` (randomly initialized critic, the best config
before this ablation), a `--lr` sweep around the value reused from
REINFORCE (1e-4):

| `--lr` | eval_avg(3000) | win_rate |
|---|---|---|
| 8e-4 | -13.96 | 48.1% |
| 1e-4 (baseline, = REINFORCE) | +9.75 | 52.2% |
| **3e-5** | **+17.60** | **53.8%** |
| 1e-5 | +10.10 | 52.2% |

A **non-monotonic** relationship: `8e-4` is destructive (entropy
gradually collapses from 0.19 to 0.05 over the run, the policy converges
too fast onto a mediocre response); `1e-5` falls back to baseline level
(stable entropy, but probably underlearned -- not enough cumulative
movement over 50k episodes with such small steps); `3e-5` is a clear sweet
spot, which **beats REINFORCE** (`seg_50000`, +15.22) at the same 50k
episode budget -- this session's first PPO config to do so.

**Methodological caveat (see also remarques_rl.md point 5):** this
comparison remains asymmetric. REINFORCE's `lr=1e-4` (reused as-is in
`run_growing_pool.py`) comes from the v1 lineage's lr/entropy ablation
(`rl_experiments/README.md`), done **before** the scoring rule-5 fix --
which raised the reward's empirical standard deviation from ~124 to ~228
points/donne (a much noisier signal since). Nothing guarantees this lr is
still optimal for REINFORCE under the new reward scale, exactly as we just
found that the lr "reused" from REINFORCE wasn't optimal for PPO (3e-5
clearly better than 1e-4). Concluding "PPO beats REINFORCE" from this
table would be premature until REINFORCE has received the same lr/entropy
ablation under the post-rule-5 signal -- to do before any definitive
conclusion.

### Consolidated results (`eval_policy.py`, 3000 games, seed=42)

| Checkpoint | eval_avg(3000) | win_rate |
|---|---|---|
| **exp_ppo_round1_lr3e-5** (best PPO so far) | **+17.60** | **53.8%** |
| seg_50000 (REINFORCE, reference -- lr not revalidated, see above) | +15.22 | 53.2% |
| exp_ppo_round1_lr1e-5 | +10.10 | 52.2% |
| exp_ppo_round1_epochs8 (lr=1e-4) | +9.75 | 52.2% |
| exp_ppo_round1_epochs4 | +7.84 | 51.8% |
| heuristic (structural-noise reference) | +7.38 | 51.8% |
| exp_ppo_round1_valuepretrain (pretrained critic, buggy target) | +4.92 | 51.2% |
| exp_ppo_round1_valuepretrain_v3 (pretrained critic, fixed target) | -0.34 | 50.3% |
| imit.pt (starting point) | -3.08 | 49.6% |
| exp_ppo_round1_lr8e-4 | -13.96 | 48.1% |

At 50k episodes and `--opponent heuristic` alone, PPO (`lr=3e-5`) beats
REINFORCE at the same budget for the first time -- but see the
methodological caveat above before drawing a firm conclusion. Possible
next steps: redo the lr/entropy ablation on REINFORCE under the
post-rule-5 signal (a fair comparison); fine-tune around `lr=3e-5` (e.g.
2e-5, 4e-5) or other hyperparameters (`clip-eps`, `batch-episodes`); still
without critic pretraining for now (idea on hold, see above).
