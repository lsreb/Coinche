# RL checkpoints - lineage v4 (CardNetBig: a deeper MLP)

A directory separate from `rl_experiments_v3/`, created when
`coinche/rl_agent.py` gained `CardNetBig` (167 -> 128 -> 128 -> 64 -> 32,
GELU, Pre-LN LayerNorm before each linear layer, ~1.9x more parameters
than `CardNet`) -- the same principle as the v1->v2 (scoring rule 5) and
v2->v3 (new `_points_so_far` state feature) transitions: the architecture
changes shape (`fc1` has a different shape), so checkpoints are no longer
compatible with `CardNet` and the previous lineages. The state/scoring
don't change (still 167-dim, same game engine): `CardNetBig` stays
loadable via `--architecture big` (`train.py`/`pretrain.py`/
`eval_policy.py`/`eval_matchup.py`, cf. `PPOPolicy.policy_net_cls`/
`NeuralPolicy.net_cls`), `CardNet` (`--architecture small`, default) stays
unchanged, and `rl_experiments_v3/`'s checkpoints stay loadable and
exactly reproducible.

Context/full plan: `coinche/remarques_rl.md` point 6. The growing pool
(`rl_experiments_v3/README.md`) is the only structural idea that held up
this session; every critic-side idea failed because `ValueNet` is a
network entirely independent of `CardNet` -- nothing can reach the actor.
Step 1 of the plan (this one): test whether capacity alone (`CardNet` has
only a single 128-unit hidden layer) is a limiting factor, isolated from
everything else (networks still separate, plain training against
heuristic, REINFORCE first).

## `imit_bignet.pt`: clearly better imitation accuracy

Same protocol as `rl_experiments_v3/imit.pt` (5000 donnes, 20 epochs,
seed 0): **98.2% imitation accuracy, vs 92.8% for `imit.pt`** (`CardNet`)
-- a strong signal that the small network's capacity was limiting how
well it could imitate the heuristic.

## lr sweep, REINFORCE, seed20 (100k episodes)

REINFORCE fine-tuning from `imit_bignet.pt` (`--architecture big`),
otherwise the same protocol as `rl_experiments_v3/exp_reinforce_lr3e-5`
(`entropy-beta=0.002, entropy-decay=none, opponent=heuristic`):

| lr | eval_avg(45000) | eval_avg(10000, different sample) |
|---|---|---|
| `3e-5` | +11.51 | +15.22 |
| `1e-4` | +14.06 | +16.38 |
| `3e-4` | -27.9 (eval_avg(500) at the end of training) | **-28.91 (policy destroyed)** |

Per-game std (on the n=10000 sample) ~228.5, standard error ±2.29 at
n=10000 -- the gap between the two measurements of the SAME lr=3e-5
checkpoint (+11.51 then +15.22 depending on the donne sample) illustrates
the residual measurement noise even at n=45000 (standard error ~1.08 on
that particular sample).

**Findings (1 seed, caution warranted):**
- `lr=3e-4` clearly destroys the policy (tens of standard errors below
  both others) -- the same pathology as `8e-4` for PPO
  (`coinche/remarques_rl.md` point 5).
- `lr=3e-5` vs `lr=1e-4`: the observed gap (1.16 to 2.55 points depending
  on the sample) is within the noise (standard error ~2.29) -- no clear
  difference between the two at this stage.
- On the latest sample (n=10000), the two non-diverged configs (+15.22,
  +16.38) end up **around the plateau established for `CardNet`**
  (`rl_experiments_v3`: PPO `epochs=8` 15.13, REINFORCE `lr=3e-5` 15.15,
  over several seeds) -- neither clearly better nor clearly worse, unlike
  what the very first measurement suggested (+11.51 alone, at the time a
  single data point).

## `lr=1e-4`, n=3: stable, but slightly below the `CardNet` plateau

3 seeds (20/21/22, same protocol, `lr=1e-4`) evaluated at n=45000:

| seed | eval_avg(45000) |
|---|---|
| 20 | +14.06 |
| 21 | +14.00 |
| 22 | +12.87 |

Mean **13.64**, std **0.67** -- very stable across seeds (the same order
of magnitude as `rl_experiments_v3`'s stable configs, std 0.39-0.40). But
~1.5 point **below** both already-established `CardNet` plateaus (PPO
`epochs=8`: 15.13/std=0.39; REINFORCE `lr=3e-5`: 15.15/std=0.40). A
Welch's t-test against each of the two: t≈3.3-3.5 (df≈3.1-3.2, critical
threshold ≈3.18) -- just above the conventional threshold, so a barely
significant gap at this stage, but not massive (n=3 against n=3-4, as
always in this lineage to be taken with caution before treating it as settled).

**Explicit scope of this conclusion (important):** this result only
concerns **`lr=1e-4` on plain REINFORCE**, not `CardNetBig` in general:
- `lr=3e-5` has never been tested over several seeds (a single point,
  +11.51 then +15.22 depending on the eval sample -- itself within
  sampling noise, see above) -- it's possible `3e-5` is actually better
  than `1e-4` for this deeper network, once confirmed over several seeds.
- This step only tests raw capacity, on REINFORCE, against heuristic
  alone, with the same imitation/data protocol as `CardNet`. It remains to
  be seen how this network learns with other data changes (more
  donnes/epochs for imitation, a growing pool...) and especially with step
  2 of the plan (`coinche/remarques_rl.md` point 6): a shared
  actor/critic trunk + centralized critic + auxiliary tasks, never
  tested -- this first isolated step doesn't prejudge the result once the
  critic is involved.

## `lr=3e-5`, n=2 (partial): wider variance, no conclusion yet

Seed21 added (same protocol, `lr=3e-5`):

| seed | eval_avg(45000) |
|---|---|
| 20 | +11.51 |
| 21 | +15.37 |

Mean 13.44 (n=2) -- close to `lr=1e-4`'s mean (13.64, n=3), but with a
much wider gap between the two seeds (3.86 points, vs 1.19 across
`lr=1e-4`'s 3 seeds) -- not yet enough seeds to say whether `lr=3e-5` has
a genuinely larger variance or if it's just n=2. Seed21 alone (+15.37)
lands right in the usual `CardNet` plateau (~15.1-15.2). **No conclusion
on which of the two lrs is better at this stage** -- a 3rd seed at
`lr=3e-5` would be needed for an equally rigorous comparison.

## Open idea: is the imitation-derived policy too confident?

Hypothesis tested (2026-07-26 discussion): `CardNetBig`'s slight
underperformance could come from an imitation starting point too
confident/inflexible for REINFORCE fine-tuning (a noisy policy gradient,
a single update per donne), rather than a capacity or vanishing-gradient
issue (not very plausible here: a shallow network, GELU + Pre-LN LayerNorm
already designed to avoid that, and the training trajectories show real
movement, not stagnation).

**Test A (verified)**: the policy's entropy right after imitation (before
any RL), over a sample of 16000 decisions (500 heuristic x4 donnes):

| | mean entropy | mean top1_prob |
|---|---|---|
| `CardNet` (`imit.pt`) | 0.2153 (std=0.3186) | 91.5% |
| `CardNetBig` (`imit_bignet.pt`) | **0.0848** (std=0.1972) | **96.6%** |

**Confirms the hypothesis**: `CardNetBig` comes out of imitation with
~2.5x lower entropy -- a much more peaked distribution (96.6% of the mass
on the preferred move, vs 91.5%). Consistent with a too-confident starting
point, even though this doesn't yet prove causality with final RL
performance.

**Next to test**: `entropy-beta=0.006` (a factor of ~2.5-3x the current
0.002, calibrated on the measured entropy gap -- preferred over an
arbitrary jump like 0.02, out of caution: this network's sensitivity to
this coefficient is unknown, and raising it too much could just as well
overcorrect into a too-random policy). Plan: first measure the
post-training entropy of the 5 checkpoints already trained at 0.002
(reusing existing checkpoints, no new training needed) as a reference,
then launch a short run (20000 episodes) at 0.006 to check the realized
entropy rises as expected, before investing in a full 100k-episode
comparison.

## Stronger entropy bonus: doesn't work -- the problem comes from imitation, not RL

Reference (post-training, `entropy-beta=0.002`, the same 5 100k-episode
checkpoints as above): entropy 0.0838-0.1173, very close to the imitation
starting point (0.0848) -- REINFORCE training barely moves the entropy.
For comparison, `CardNet` (v3) stays its whole life within 0.17-0.22
(imitation as much as after RL) -- `CardNetBig` operates at roughly half
that level, at every stage.

Tested a higher `entropy-beta` to compensate, on SHORT runs (20000
episodes, same seed20, `lr=1e-4`, to isolate the coefficient's effect at
equal training volume):

| entropy-beta | mean entropy (20000 ep) | mean top1_prob |
|---|---|---|
| `0.002` (reference) | 0.0966 | 96.10% |
| `0.006` (~3x) | 0.0986 | 95.98% |
| `0.06` (~30x) | **0.1018** | 95.91% |

**The entropy bonus has almost no traction here**: even multiplied by
30x, the entropy barely moves (+0.005 relative to the reference).
Hypothesis: near an almost-deterministic distribution (like
`CardNetBig`'s right out of imitation), the entropy's gradient with
respect to the logits becomes very flat -- no reasonable coefficient can
move it significantly once the policy is this confident. The problem
therefore comes from **imitation itself**, not from a fix to apply
afterward during RL.

## Stopping imitation at a target entropy (0.20, `CardNet`'s level)

Rather than fixing it after the fact, imitation training is stopped as
soon as the policy's entropy (measured on the same 500-donne reference
sample) reaches `CardNet`'s target (~0.20), instead of running to full
convergence (20 epochs, where entropy drops to 0.0848).

| epoch | accuracy | entropy (probe) |
|---|---|---|
| 1 | 82.1% | 0.3827 |
| 2 | 89.0% | 0.2933 |
| 3 | 90.9% | 0.2421 |
| 4 | 92.1% | 0.2149 |
| **5** | **92.9%** | **0.1922** (target reached) |

Stopped at epoch 5 -> `rl_experiments_v4/imit_bignet_ent02.pt`. Notable
fact: the accuracy at this point (92.9%) is nearly identical to fully
converged `CardNet`'s (92.8%) -- both networks reach the same level of
imitation at a comparable entropy, `CardNetBig` just much faster (5
epochs vs 20) thanks to its extra capacity. Same architecture as
`imit_bignet.pt` (no compatibility break, stays within this v4 lineage,
not a new lineage -- only the training stopping point changes, not the
weights' shape).

## REINFORCE from `imit_bignet_ent02.pt` (lr=3e-5): a massive gain, CONFIRMED at n=3

Exactly the same protocol as the previous runs (`lr=3e-5,
entropy-beta=0.002, entropy-decay=none, opponent=heuristic, 100k
episodes`), but from `imit_bignet_ent02.pt` (entropy 0.192, epoch 5)
instead of `imit_bignet.pt` (entropy 0.0848, epoch 20).

| seed | eval_avg(45000) |
|---|---|
| 20 | +25.04 |
| 21 | +23.21 |
| 22 | +25.00 |

Mean **24.42**, std **1.05** (n=3) -- very tight, a real signal, not
noise. For comparison: the established `CardNet` plateau (PPO `epochs=8`:
15.13/std=0.39; REINFORCE `lr=3e-5`: 15.15/std=0.40, both over several
seeds); `CardNetBig` from the old `imit_bignet.pt` (imitation to full
convergence) at `lr=3e-5` (n=2): 11.51, 15.37 (mean 13.44, borderline
worse than `CardNet`).

**A delta of +9.27 points relative to `CardNet`'s REINFORCE `lr=3e-5`
plateau** -- a Welch's t-test gives t≈14.6, tens of standard errors above
any significance threshold. By far this whole session's largest and
best-confirmed result (REINFORCE or PPO, v3 or v4).

**Conclusion**: the full hypothesis held. `CardNetBig`'s extra capacity
(128/128/64, GELU, Pre-LN LayerNorm) really does help -- but only if the
too-confident starting point from imitation run to full convergence
(entropy 0.0848, where the entropy bonus during RL has no traction) is
avoided. By stopping imitation at a target entropy (~0.20, `CardNet`'s
level) instead of running to convergence, REINFORCE fine-tuning starts
from a far more exploitable point and unlocks a massive, now
well-confirmed gain (n=3, std=1.05).

## To do if continuing

- Redo `lr=1e-4` from `imit_bignet_ent02.pt` (at least 1 seed) to see
  whether the lr still matters now that the right starting point is used,
  or whether `lr=3e-5` now clearly dominates.
- Test PPO with `CardNetBig` + `imit_bignet_ent02.pt` (never done,
  REINFORCE only so far) -- given PPO≈REINFORCE everywhere else in this
  lineage, expect a similar gain, but to be verified.
- Try other entropy targets around 0.20 (e.g. 0.15, 0.25) to see whether
  the exact point matters, or whether the whole "not fully converged"
  zone works the same.
- Then move to step 2 of the plan (shared actor/critic trunk + centralized
  critic, `coinche/remarques_rl.md` point 6) -- with this new imitation
  starting point now as the reference to beat.

## `lr=1e-4` from `imit_bignet_ent02.pt` (seed20): much weaker than `lr=3e-5`

Same protocol, only the lr changes: **eval_avg(45000) = +11.81** -- well
below the confirmed `lr=3e-5` mean (24.42, std=1.05 over 3 seeds). The
gap (~12.6 points) is far beyond the inter-seed noise observed at
`lr=3e-5`, so probably not just an unlucky seed: `lr=3e-5` appears
specifically necessary to exploit this new starting point, unlike small
`CardNet`, where `lr=3e-5` and `lr=1e-4` were statistically
indistinguishable. Only one seed for now at `lr=1e-4` -- no more seeds
planned yet, this point is secondary compared to `lr=3e-5`'s n=3
confirmation.

## Step 2 of the plan: shared actor/critic trunk + centralized critic

Implemented (not yet tested under real conditions): `SharedTrunkActorCritic`
(`coinche/rl_agent.py`) and `SharedTrunkPPOPolicy` (`train_ppo.py`,
`--architecture shared`).

Architecture (2026-07-27 discussion):
- **Shared trunk** (actor + critic): `CardNetBig`'s first 2 layers
  (`167->128->128`, GELU + Pre-LN LayerNorm) -- same parameter names as
  `CardNetBig` so an already-trained `imit_bignet_ent02.pt` can be reused
  as a starting point (`SharedTrunkActorCritic.load_actor_from_cardnetbig`,
  verified bit-for-bit identical to `CardNetBig` on the same checkpoint).
- **Actor head**: continues on its own, `128->64->32` -- identical in
  shape/names to `CardNetBig`'s second half, the policy's behavior
  unchanged at equal weights.
- **Critic head**: additionally receives the centralized info (the 3
  others' hands, 96-dim, already in `encode_full_state`) through a single
  separate layer (`96->32`, a deliberate choice: a more structural than
  strategic role, 2026-07-27 discussion), concatenated to the shared
  trunk's output (`128+32=160`) before 1 private hidden layer (`160->64`)
  then the scalar output.
- No auxiliary tasks in this first version (2026-07-27 decision: isolate
  the shared trunk's effect alone before adding this complexity).

The key point relative to the 4 previous critic attempts (all with a
`ValueNet` entirely independent of `CardNet`): here `value_loss`'s
gradient also flows through the shared trunk, so the centralized info can
finally influence the representation the actor uses -- a mechanism every
previous attempt lacked by construction.

Smoke-tested end to end (loading from `imit_bignet_ent02.pt`, 200 training
episodes, save/reload, `eval_policy.py --architecture shared`) -- no errors.

### First real runs: value/policy interference through the trunk, then a fix

First full run (100k episodes, `lr=3e-5`, `epochs=8, value-coef=0.5` --
the same PPO hyperparameters as the `CardNet` reference) from
`imit_bignet_ent02.pt`: **eval_avg(45000) = +12.58**, clearly below the
REINFORCE reference (24.42, n=3) and even slightly below the plain
`CardNet` plateau (~15.1-15.2). Training trajectory clearly noisier than
usual (`clip_frac` 5-6%, vs <2% everywhere else this session) -- a sign of
value/policy interference through the shared trunk, exactly the
historical problem that had led to abandoning the original PPO
architecture (`ActorCriticNet`, before `policy_net`/`value_net` became
independent).

Exploratory sweep (seed20 for all, `lr=3e-5` fixed) to fix this:

| epochs | value_coef | eval_avg(45000) | clip_frac (end of run) |
|---|---|---|---|
| 8 | 0.5 | +12.58 | 5.2% |
| 4 | 0.5 | +20.53 | 1.9% |
| 4 | 0.3 | +23.81 | 1.6% |
| 4 | 0.1 | +24.42 | 1.0% |
| **4** | **0.2** | **+27.80** | 2.2% |

Both factors matter: going from `epochs=8` to `epochs=4` (with
`value_coef` unchanged) already covers half the gap (12.58 -> 20.53,
`clip_frac` back to a normal range) -- with a shared trunk, every pass
updates the representation from both losses, so 8 passes seems to
over-perturb the trunk. Lowering `value_coef` on top adds a further gain,
with an apparent optimum around 0.2 (0.1 and 0.3 give close but slightly
lower results).

### Confirmation at n=4 of the best config (`epochs=4, value_coef=0.2`)

| seed | eval_avg(45000) |
|---|---|
| 20 | +27.80 |
| 21 | +22.09 |
| 22 | +25.23 |
| 23 | +23.28 |

Mean **24.60**, std **2.50** (n=4) -- to compare against REINFORCE from
the same `imit_bignet_ent02.pt` (mean 24.42, std 1.05, n=3). **The two
means are nearly identical** (a 0.18-point gap): the shared trunk matches
the best known config, without beating it, with wider inter-seed variance
(2.50 vs 1.05).

**Conclusion**: the mechanism works -- unlike the 4 previous critic
attempts (all with an independent `ValueNet`), here sharing the trunk
doesn't hurt performance once the hyperparameters are correctly
recalibrated for this architecture (fewer epochs, lower `value_coef`) --
but no proof of a net gain over plain REINFORCE from the same starting
point, even at n=4. Next idea (cf. the 2026-07-27 discussion,
`rl_experiments_v5/`): the centralized critic has a shortcut (the
centralized-info branch) that dilutes its own gradient on the shared
trunk -- an auxiliary task placed SOLELY on the trunk (without that
shortcut) might have a stronger effect, worth testing separately.

**To do if continuing**:
- More seeds at `epochs=4, value_coef=0.2` to settle whether the shared
  trunk actually beats REINFORCE or just matches it.
- Try other `value_coef` values around 0.2 (e.g. 0.15, 0.25) with several
  seeds, since a single seed per point can't distinguish a real optimum
  from noise.
- Add the auxiliary tasks (remaining trumps/aces of the 3 others, the
  plan's initial discussion) now that the shared trunk alone works at
  least as well as REINFORCE.

## Growing pool from `imit_bignet_ent02.pt`: this session's best result (seed40, n=1)

After the critic/auxiliary idea (shared trunk, then the auxiliary task in
`rl_experiments_v5/`) produced nothing clearly beyond plain REINFORCE,
back to this session's other confirmed structural result: the growing
self-play PFSP pool (`rl_experiments_v3/README.md`, n=3 REINFORCE + n=2
PPO, never tested with `CardNetBig`/entropy-targeted imitation). The two
gains had never been stacked.

`run_growing_pool.py --architecture big --init-load imit_bignet_ent02.pt
--lr 3e-5 --total-episodes 300000 --segment-episodes 50000 --seed-start 40`
(6 segments, `rl_experiments_v4/exp_growing_pool_bignet/`). Fixed a
latent bug along the way in `train.py` (`make_opponent_factory`/
`build_opponent_pool` always instantiated `NeuralPolicy(net_cls=CardNet)`
for a frozen pool opponent, never an issue as long as the pool only
contained `CardNet` checkpoints -- `load_state_dict` would have failed
starting at segment 2 with a `CardNetBig` pool). All 6 segments ran
cleanly (18.5 to 35 min each, slightly rising with the pool size; PFSP
samples well across every pool member at healthy 0.49-0.60 win rates, no
sign of collapse).

**eval_avg(45000) = +31.70**, win_rate=56.3% -- above the fixed REINFORCE
reference from the same starting point (+25.04 for this exact seed, mean
24.42/n=3) and the shared trunk (24.60, n=4). **The best result obtained
in this whole RL session.**

**Only one seed for now** -- the gap (~6-7 points above the best
references) exceeds the usual inter-seed variance (std 1.05 to 2.50
depending on the architecture), so it's promising, but not yet confirmed.
**To do before concluding**: at least 2 more seeds (`--seed-start 41` and
`42` for instance, same hyperparameters) to confirm the gain holds, the
same rigor as everywhere else this session.

### Round-robin (`eval_matchup.py`, n=10000): the gain is confirmed, not just an `eval_avg` artifact

The same approach as validating the growing pool in v3 (cf.
`rl_experiments_v3/README.md`): `eval_avg` against heuristic alone only
measures exploitation of a fixed opponent, not versatility. Added
`--architecture` to `eval_matchup.py` (until now it only supported a
hardcoded `PPOPolicy()`/small `CardNet` -- the same convention as
`eval_policy.py`) so it can be used on `CardNetBig` checkpoints.

4 specs, 6 matchups (n=10000, seed=42): `heuristic`, this lineage's
"vanilla" reference (`exp_reinforce_bignet_ent02_lr3e-5_seed20/
reinforce_100000.pt`, +25.04 in `eval_avg`), the pool's early checkpoint
(`seg_50000`), and the final checkpoint (`seg_300000`).

Average strength (seat-bias-corrected, `skill(X,Y) = ((X vs Y) -
(Y vs X))/2`, the same method as in v3):

| | average strength |
|---|---|
| **bignet pool, seg_300000 (final)** | **+18.97** |
| vanilla (plain bignet REINFORCE) | +6.43 |
| bignet pool, seg_50000 (early) | +0.31 |
| heuristic | -25.71 |

**The same pattern as in v3, without exception, and with a sharper gap**:
the pool's final checkpoint clearly beats vanilla head-to-head (+15.88 /
-6.38, i.e. +11.13 once seat-bias corrected) and beats the early
checkpoint by a wide margin (+20.08 / -9.52, i.e. +14.8 corrected); the
early checkpoint itself doesn't clearly stand out from vanilla (+0.31 vs
+6.43, on the same order as the variance observed in v3 between similar
configs). The +31.70 `eval_avg` gain is therefore not (only) an artifact
of this noisy metric -- it's confirmed head-to-head, as for v3's "small
CardNet" growing pool.

**Still n=1**: remains to be confirmed on 2 more seeds (as for v3, where
the pattern ended up confirmed identically over 3 REINFORCE seeds + 2 PPO
seeds) before treating this result as definitively established.

## `--no-critic-baseline` on the shared trunk (seed20): still no net effect

Unlocked `--no-critic-baseline` for `--architecture shared` (previously
forbidden by the code): advantage = return centered/normalized on the
batch (REINFORCE-style), the critic head is neither called nor trained
(`value_loss` stays at 0 throughout -- verified). Question asked: is the
shared trunk the ONLY architecture where the critic has a real gradient
channel to the actor (unlike the 4 previous attempts with an independent
`ValueNet`, all with no measurable effect) -- does its baseline actually
serve a purpose this time?

Exactly the same protocol as the reference (`lr=3e-5, epochs=4,
value_coef=0.2` -- the latter no longer has any effect here since
`value_loss=0`), **the same seed20** as the already-known reference, for
a direct comparison:

| | eval_avg(45000) |
|---|---|
| seed20, with critic (`value_coef=0.2`) | +27.80 |
| seed20, **without** critic (`--no-critic-baseline`) | +24.03 |

Raw seed-to-seed comparison: -3.77 points, looks like a degradation.
**But read against the true distribution of the 4 already-confirmed
with-critic seeds (22.09 / 23.28 / 25.23 / **27.80**, mean 24.60, std
2.50)**, +24.03 lands comfortably WITHIN that range, very close to the
mean -- and `27.80` (seed20 with critic) happens to be the HIGHEST of the
4 known seeds, not a typical point. Comparing "no critic" specifically to
that one seed (the most favorable to the critic) rather than to the
group's mean biases the reading toward "the critic helps" when nothing
clearly shows that once inter-seed variance is accounted for.

**Honest reading (n=1 for the no-critic config, caution warranted)**: no
proof the critic brings a net gain on the shared trunk either -- **this
project's 7th consecutive result of this kind** (after 4 attempts with an
independent `ValueNet` + the shared trunk itself, which only matches
REINFORCE): the critic's baseline shows its usefulness nowhere in this
code, even where it finally has a real channel to the actor. Continuing
would require 2-3 more seeds at `--no-critic-baseline` for a real
statistical test against the existing n=4 distribution -- not done yet, a
later idea.

## `SharedTrunkActorCriticDeep` (3-layer trunk, 1-layer actor head): first test, seed20

A rebalancing of `SharedTrunkActorCritic` (the user's exact design,
2026-07-28 discussion): a deeper trunk (`167->128->128->64`, 3 layers --
covering EXACTLY `CardNetBig`'s first 3 layers) and a shorter actor head
(`64->32`, a single layer -- `CardNetBig`'s last one), instead of the
current 2+2 split. Critic head: a centralized branch `96->64` (equal to
the trunk's output dim, a more balanced merge than the old 128 vs 32), 1
private layer `128->32`, output `32->1`.

An independent class (`coinche/rl_agent.py`), `SharedTrunkActorCritic`
never modified in place. `load_actor_from_cardnetbig` remaps
`imit_bignet_ent02.pt` with no loss at all (the trunk/head split matches
exactly `CardNetBig`'s 4 layers) -- verified bit-for-bit identical, so
**no need to retrain imitation**. `SharedTrunkPPOPolicy` gains a `net_cls`
parameter (like `PPOPolicy`) instead of a duplicated wrapper class. New
`--architecture shared_deep` choice.

Exactly the same protocol as the reference (`lr=3e-5, epochs=4,
value_coef=0.2`, from `imit_bignet_ent02.pt`, seed20 -- the same seed as
the 2 other already-tested variants, for a direct comparison). Training
healthy throughout (normal `clip_frac`, no collapse).

| | eval_avg(45000) |
|---|---|
| Original trunk (2+2), seed20, with critic | +27.80 |
| Original trunk (2+2), seed20, without critic | +24.03 |
| **Deep trunk (3+1), seed20, with critic** | **+23.70** |
| Original trunk, mean n=4 | 24.60 (std 2.50, range 22.09-27.80) |
| Plain REINFORCE, mean n=3 | 24.42 (std 1.05) |

**Honest reading (n=1)**: `+23.70` lands comfortably within the original
trunk's already-established distribution (22.09-27.80), very close to its
mean (24.60) and REINFORCE's (24.42). Compared specifically to the
original seed20 (27.80) it looks worse, but that seed is the highest of
the 4 known ones (the same reading trap as for `--no-critic-baseline`
above) -- not a fair comparison. **No proof that rebalancing trunk/head
changes anything**, either way. To be confirmed over more seeds if
settling it matters, but nothing urgent given this first neutral point.

### Seed21: confirmed, very tight around the same neutral point

Exactly the same protocol, seed21: **eval_avg(45000) = +22.78**.

| seed | eval_avg(45000) |
|---|---|
| 20 | +23.70 |
| 21 | +22.78 |

Mean **23.24**, std **0.65** (n=2) -- very tight, the two seeds nearly
indistinguishable from each other. Confirms seed20's reading alone:
`SharedTrunkActorCriticDeep` lands right within the original trunk's
distribution (22.09-27.80, mean 24.60, n=4) and near REINFORCE (24.42,
n=3), with no sign of improvement or degradation. Rebalancing trunk/head
(3 shared layers + 1 private, instead of 2+2) therefore doesn't seem to
change anything, at least at this test scale -- a neutral idea, not a
priority for now compared to the growing pool, which remains the only
direction to have clearly beaten this plateau.

## Growing pool + shared trunk (original, seed-start 60): no net gain, unlike the bignet run

After the growing pool + `CardNetBig`/REINFORCE (+31.70, this session's
best result), the same experiment with the shared trunk
(`SharedTrunkActorCritic` original, 2+2, `value_coef=0.2` -- not the
`Deep` variant, deliberately chosen to isolate a single variable at a
time against an already well-characterized architecture, n=4).

`run_growing_pool_ppo.py --architecture shared --init-load
imit_bignet_ent02.pt --lr 3e-5 --epochs 4 --value-coef 0.2
--total-episodes 300000 --segment-episodes 50000 --seed-start 60` (6
segments, `rl_experiments_v4/exp_growing_pool_shared/`). All segments ran
cleanly (1151-1937s, the same gradual rise as elsewhere; also confirmed
along the way that the `SharedTrunkActorCritic.forward()` fix for loading
a frozen `shared` opponent in the pool works under real conditions, not
just in the smoke test).

| | eval_avg(45000) |
|---|---|
| Plain shared trunk (seed20, no pool) | +27.80 |
| Pool, early (seg_50000) | +19.85 |
| **Pool, final (seg_300000)** | **+25.21** |
| Plain shared trunk, mean n=4 | 24.60 (std 2.50, range 22.09-27.80) |

**First reading (via `eval_avg` alone)**: compared to the exact seed20
(27.80), the pool's final checkpoint looks like a regression -- but 27.80
is again the highest of the 4 known seeds (the same trap as this
session's 2 previous readings). Compared to the full distribution
(22.09-27.80, mean 24.60), +25.21 lands right within it, essentially at
the mean: `eval_avg` alone shows **no net gain** over the shared trunk
without a pool.

### Round-robin (`eval_matchup.py`, n=10000): the gain does exist, invisible in `eval_avg` alone

The same approach as for the bignet+REINFORCE run: 4 specs (heuristic,
plain shared trunk seed20, pool early `seg_50000`, pool final
`seg_300000`), n=10000, seed=42. Average strength (seat-bias-corrected):

| | average strength |
|---|---|
| **Pool, final (seg_300000)** | **+15.45** |
| Plain shared trunk (no pool) | +9.00 |
| Pool, early (seg_50000) | -0.11 |
| heuristic | -24.34 |

**The same exceptionless pattern as in v3 and as for the bignet run**:
final > plain without pool > early > heuristic. The final pool checkpoint
beats the plain version head-to-head (+10.00 / -4.72, i.e. +7.36
corrected) and crushes the early pool checkpoint (+16.51 / -11.11, i.e.
+13.81 corrected); the plain version itself beats the early one (+9.13 /
-3.43, i.e. +6.28 corrected).

**This contradicts the first reading based on `eval_avg` alone**: there
is indeed a real versatility gain (+6.45 points of corrected strength
between pool-final and plain), on the same order as the gain seen for the
bignet pool -- simply invisible against heuristic alone, exactly v3's
lesson (`eval_avg` against a single fixed opponent only measures
exploitation of that opponent, not real versatility). So the growing pool
does bring a gain with the shared trunk/PPO too, contrary to what the
first reading suggested -- **only one seed for now**, to be confirmed
over more seeds before concluding definitively, the same rigor as
everywhere else this session.

## Direct head-to-head: bignet+REINFORCE pool vs shared-trunk+PPO pool (n=100000)

This session's two best results (growing pool + `CardNetBig`/REINFORCE,
`eval_avg`=+31.70; growing pool + shared trunk/PPO, `eval_avg`=+25.21 but
round-robin strength +15.45) had never been pitted directly against each
other -- each only compared to heuristic and to its own reference group.
Added mixed-architecture support to `eval_matchup.py` (per-spec
`architecture:path` syntax, e.g. `big:seg_300000.pt shared:seg_300000.pt`,
previously a single `--architecture` for the whole call) to make this
test possible.

A 3-spec round-robin (heuristic, `big:exp_growing_pool_bignet/seg_300000.pt`,
`shared:exp_growing_pool_shared/seg_300000.pt`), **n=100000** games per
matchup (this whole session's finest precision, ~10x the usual
`--games 45000`):

| | avg |
|---|---|
| bignet-pool vs shared-pool | -0.17 |
| shared-pool vs bignet-pool | +1.06 |

Seat-bias corrected: `skill(bignet, shared) = (-0.17 - 1.06)/2 =
-0.615`. **Statistically null at this scale** (n=100000/direction, this
whole session's greatest precision): the two checkpoints are tied
head-to-head, despite very different `eval_avg` against heuristic (+31.70
vs +25.21) and different round-robin strength scores in their respective
groups (+18.97 vs +15.45 -- but not computed against the same reference
groups, so never directly comparable to each other before this test).

**Conclusion**: the growing pool brings a real gain in both cases
(separately confirmed by round-robin for each architecture), but once
combined with the pool, the choice of base architecture (REINFORCE+
CardNetBig vs PPO+shared trunk) no longer seems to make a difference in
final level -- both converge to a similar playing strength.

## Bignet+REINFORCE pool extended to 450k (segments 7-9): `eval_avg` declines, round-robin needed to settle it

Same seed (40, `--start-segment 7` on `rl_experiments_v4/exp_growing_pool_bignet/`,
same hyperparameters) to see whether progress continues past 300k. The 3
extra segments ran cleanly (2072-2108s each, within the norm).

| Segment | eval_avg(45000) |
|---|---|
| 300k | +31.70 |
| 350k | +31.28 |
| 400k | +29.53 |
| 450k | +28.70 |

**Cautious reading**: `eval_avg` against heuristic alone gradually
declines past 300k -- but this metric alone is known to be misleading for
a pool-trained checkpoint (v3's lesson, already confirmed twice this
session: the shared-trunk growing pool's "real gain" was invisible in
`eval_avg`, revealed only by round-robin).

### Round-robin (n=10000): the "decline" doesn't hold up head-to-head

4 specs (heuristic, vanilla REINFORCE, pool 300k, pool 450k):

**Direct head-to-head 300k vs 450k**: 300k vs 450k = -1.60, 450k vs 300k
= +8.56 -- corrected, **450k beats 300k by +5.08**. Head-to-head, 450k is
therefore NOT weaker than 300k, quite the opposite.

Average strength (against the other 3):

| | average strength |
|---|---|
| Pool, 300k | +15.73 |
| Pool, 450k | +11.40 |
| Vanilla (plain REINFORCE) | +1.51 |
| Heuristic | -28.63 |

**An apparent tension, but not a contradiction**: in average strength,
300k appears to be ahead -- driven by its better performance against
heuristic and vanilla (weaker opponents), a well-known artifact of this
kind of small-field ranking (a mean over a small group of opponents isn't
necessarily transitive with the direct head-to-head between the two
best). The direct head-to-head remains the most relevant signal to answer
the question asked ("is 450k weaker than 300k?"): the answer is no, if
anything the opposite.

**Conclusion (corrected)**: `eval_avg`'s "decline" doesn't reflect a real
regression -- yet another specialization shift (less exploitation of
heuristic specifically). But unlike an overly cautious first reading
("essentially tied"), the head-to-head shows a **real gain**, not just an
absence of loss: +5.08 corrected, consistent in both directions (300k
loses in both match orientations), on the same order as this session's
other gaps treated as real at n=10000 (+6.45, +7.36, +11.13...).
Continuing training up to 450k therefore genuinely helped. Open question:
does it keep progressing past 450k, or is that where it truly plateaus?
Not tested yet.

## Extension to 600k (segments 10-12): progress continues, remarkably linearly

Same seed (40), same hyperparameters, `--start-segment 10` from
`seg_450000.pt`. 3 more healthy segments (1963-2185s each).

Round-robin (4 specs: heuristic, vanilla REINFORCE, pool 450k, pool 600k,
n=10000):

**Direct head-to-head 450k vs 600k**: 450k vs 600k = -2.64, 600k vs 450k
= +7.95 -- corrected, **600k beats 450k by +5.30**. Nearly the same gap
as 450k vs 300k (+5.08)!

Average strength (against the other 3):

| | average strength |
|---|---|
| **Pool, 600k** | **+19.57** |
| Pool, 450k | +11.33 |
| Vanilla (plain REINFORCE) | -0.81 |
| Heuristic | -30.09 |

**A perfectly monotonic ranking** (600k > 450k > vanilla > heuristic),
and the gap between each 150k-episode step is remarkably stable (~+5
points each time, 300k->450k and 450k->600k). No sign of a ceiling -- on
the contrary, progress that looks like a straight line. `eval_avg` also
keeps suggesting a "decline" relative to 300k for these giant checkpoints
against heuristic alone (specialization shift, see above), so it
shouldn't be used to judge this trend -- only the round-robin allows that.

**To do if continuing**: push further still (750k, 900k...) to see how
far progress continues before truly plateauing.

## `legal_moves()` bug in TA (2026-07-29): fixed, impact assessed, no retrain needed

Found while playing an interactive game (`play_interactive.py`):
`legal_moves()` (`coinche/game.py`) treated TA exactly like SA -- no
obligation to overtrump when following the suit led, even though
`regles_coinche.md` explicitly says that in TA "the order and values of
all suits are those of trump" (the same logic as trump led in a suit
contract). Fixed (commit `7849f42`): TA now reuses the `lead_suit ==
trump` branch's logic (an obligation to overtrump if possible, otherwise
play a lower card of the suit -- never a free discard as long as that
suit is held). SA unchanged. Verified on 3 scenarios + 500 simulated
heuristic x4 games (51 in TA) with no crash.

**Impact on past training data, assessed before deciding on a retrain**:
`HeuristicPlayer._follow_card()` already computes the cards that beat the
trick itself and ALWAYS plays one as a priority when possible,
independently of what `legal_moves()` allowed -- its strategy already
complied with the correct rule through its own logic. Heuristic-vs-heuristic
games (the imitation targets, and the opponent throughout RL training)
were therefore barely affected at all. Only the RL policy could
technically have sampled a now-illegal card during exploration -- but
imitation had already learned to reproduce the heuristic's "always
overtrump", so the probability on those moves was already near zero even
before the fix. **Conclusion: no need for a full retrain.** This
session's relative comparisons remain valid (everything measured under
the same rule, end to end, round-robins included).

**Note for the next session**: redo an evaluation round-robin
(sanity-check under the fixed rule) and keep pushing the growing pool
further (cf. the previous section, no sign of a ceiling at 600k).

## Extension to 750k (segments 13-15): progress continues, but clearly slows

Same seed (40), same hyperparameters, `--start-segment 13` from
`seg_600000.pt`. 3 more healthy segments (2030-2198s each).

Round-robin (4 specs: heuristic, vanilla REINFORCE, pool 600k, pool 750k,
n=10000):

**Direct head-to-head 600k vs 750k**: 600k vs 750k = -1.24, 750k vs 600k
= +2.91 -- corrected, **750k beats 600k by +2.08**. Positive, but clearly
smaller than the previous steps (+5.08 for 300k->450k, +5.30 for
450k->600k).

Average strength (against the other 3):

| | average strength |
|---|---|
| **Pool, 750k** | **+18.97** |
| Pool, 600k | +17.10 |
| Vanilla (plain REINFORCE) | -3.00 |
| Heuristic | -33.07 |

**Still monotonic, but the gap between steps has been divided by ~2.5**
(from ~+5 to ~+2). Diminishing returns starting to show -- a first sign
(not proof yet) that a real ceiling is being approached, after 3
consecutive steps of real progress (300k->450k->600k->750k).

**To do to settle it**: at least one more step (900k) to see whether the
gap keeps shrinking down to zero, or stabilizes at a small residual
positive gain.

## Re-verification after the TA fix: the 300k/450k/600k round-robins hold up (out of curiosity)

The 300k-vs-450k and 450k-vs-600k round-robins above were computed
BEFORE the `legal_moves()` TA bug fix (fixed only at the time of the
interactive game, between the 600k and 750k extensions) -- only the
600k-vs-750k round-robin ran under the fixed engine. Redone here with all
3 checkpoints together (heuristic, vanilla, 300k, 450k, 600k, n=10000)
under the fixed engine, out of curiosity, to see whether the fix changes anything.

| Comparison | Before the fix | After the fix |
|---|---|---|
| 450k beats 300k | +5.08 | +4.74 |
| 600k beats 450k | +5.30 | +6.10 |
| 600k beats 300k | (never measured directly) | +10.43 |

Average strength (against the other 4, fixed engine):

| | average strength |
|---|---|
| Pool, 600k | +17.48 |
| Pool, 450k | +9.29 |
| Pool, 300k | +6.58 |
| Vanilla (plain REINFORCE) | -2.40 |
| Heuristic | -30.96 |

**A perfectly monotonic ranking, gaps nearly identical to before the
fix** (variation of ±10-15%, within normal inter-run noise). Directly
confirms what the bug's impact assessment had predicted (see the previous
section): the bug was largely inert in practice, the fix doesn't change
this experiment lineage's history.

## Extension to 900k (segments 16-18) with new PFSP settings: still a modest gain

Same seed (40), `--start-segment 16` from `seg_750000.pt`, but with the
new settings discussed the same day: `--pfsp-temperature 0.05` (was 0.1),
`--pfsp-explore-eps 0.01` (an exploration floor, new), and
`--pool-exclude 50000,150000,250000` (manually removes these 3 early
segments from the opponent pool). **`--pool-max-size` was NOT used this
time** (an omission flagged by the user) -- the pool stayed at 15 active
members (14 snapshots + heuristic), no hard cap.

PFSP distribution checked (end-of-segment log, `PFSP picks total`): the
sampling gap between the easiest (`heuristic`, ~1.4-2.4%) and the
toughest (~11-12%) is now much wider than before (4-9% at 750k without
these settings) -- the lower temperature does concentrate much more
strongly, and the floor keeps `heuristic` from dropping below ~1.4%,
close to the targeted 1%.

Round-robin (4 specs: heuristic, vanilla, pool 750k, pool 900k, n=10000):
**900k beats 750k by +2.43** corrected head-to-head -- nearly identical
to the previous step (600k->750k, +2.08, obtained WITHOUT these new
settings). Average strength still monotonic (900k +19.81 > 750k +17.47 >
vanilla -3.40 > heuristic -33.88).

**Honest reading**: the new PFSP settings (temperature/floor/manual
exclusion) didn't measurably reignite progress -- the gain stays on the
same order as without them. Two possible interpretations, not yet
settled: (1) the slowdown is a real structural ceiling of the pool at
this scale, not (only) a PFSP-dilution issue; (2) the combination tested
is incomplete, since `--pool-max-size` (the hard cap on pool size,
supposedly with the most direct impact on dilution) wasn't enabled this time.

## Extension to 1050k (segments 19-21) with `--pool-max-size 8`: the gain rebounds sharply

Same seed (40), `--start-segment 19` from `seg_900000.pt`, the same
`--pfsp-temperature 0.05`/`--pfsp-explore-eps 0.01` as before, but this
time **with** `--pool-max-size 8` (plus `--pool-exclude` extended to
`50000,100000,150000,250000`, on the user's suggestion -- also removing
`seg_100000` from the starting pool). Segment 19's starting pool computed
from the win rates measured at the previous segment: `200000, 350000,
450000, 550000, 600000, 800000, 850000, 900000` + heuristic (8 + 1,
exactly the requested cap) -- notably NOT a sort by age: `650000`/`750000`
(recent but grown easy, wr=0.60-0.66) dropped in favor of
`100000`/`200000` (older but still tough, wr~0.53) right from the initial
computation. The pool then keeps getting re-pruned by weakness at every
new segment (still 8+1, an evolving composition). 3 healthy segments
(2123-2200s each).

Round-robin (4 specs: heuristic, vanilla, pool 900k, pool 1050k,
n=10000): **1050k beats 900k by +4.28** corrected head-to-head -- more
than double the previous step (750k->900k, +2.43, WITHOUT
`--pool-max-size`), and back within the range of the "healthy" steps from
before the slowdown (+4.74, +6.10). Average strength: 1050k +21.47 > 900k
+17.57 > vanilla -4.23 > heuristic -34.81.

**Reading (still a single data point, caution warranted)**: this leans
fairly clearly toward the "PFSP dilution" hypothesis rather than
"structural ceiling" -- capping the active pool at 8 members (instead of
letting it grow without limit, as in the previous steps) seems to have
genuinely reignited progress, rather than just better targeting an
already-too-diluted pool via temperature/floor alone (tried alone at the
previous step, with no effect). To be confirmed over one more step before
concluding definitively.
