# RL checkpoints - lineage v5 (auxiliary task on the shared trunk)

A directory separate from `rl_experiments_v4/`, created not for a technical
break (the state/scoring, `imit_bignet_ent02.pt`, and the
`SharedTrunkActorCritic` architecture all stay unchanged and reusable as-is)
but because it's a substantial new experimental direction -- same logic as
`rl_experiments_v4/` for `CardNetBig` (which also didn't break the state/
scoring, just a network architecture different enough to deserve its own
lineage).

Context (2026-07-27 discussion, `coinche/remarques_rl.md` point 6):
`rl_experiments_v4/` confirmed (n=4) that the shared trunk
(`SharedTrunkActorCritic`) matches REINFORCE (24.60 vs 24.42) without
beating it. Idea identified to go further: the centralized critic has a
shortcut available (its centralized-info branch, which sees the 3 others'
hands directly) to reduce its own loss -- so its gradient on the shared
trunk is diluted, since it isn't forced to push that info up into the trunk
to improve its prediction. An auxiliary task placed on the trunk **alone**
(without that shortcut) has no choice but to improve the trunk itself to
reduce its loss -- a more direct shaping mechanism on the representation the
actor also uses.

## `SharedTrunkActorCriticAux` (`coinche/rl_agent.py`) + `SharedTrunkAuxPPOPolicy` (`train_ppo.py --architecture shared_aux`)

A subclass of `SharedTrunkActorCritic` (never modified in place) that adds
a 3rd head, predicting the number of trumps remaining for each of the 3
other seats (`other_trump_counts`), from the shared trunk `h` ALONE (not
the critic's centralized-info branch -- deliberately, to prevent the
network from "cheating" by reading the answer out of that branch instead
of encoding it into `h`). A single hidden layer (`128->32->3`, the same
depth as the critic's centralized-info branch).

New hyperparameter `--aux-coef` (the auxiliary loss's weight, MSE
normalized by its own standard deviation, same logic as `return_scale`
for `value_loss`).

**SA/TA mask**: "canonical slot 0" only corresponds to a real trump for a
suit contract (none in SA, all 4 suits tied in TA) -- decisions made during
an SA/TA contract are therefore excluded from the auxiliary loss
(`aux_valid` mask), without affecting `policy_loss`/`value_loss`, which
train normally on them. Not a bias: it only reduces the amount of data
available for the auxiliary task specifically (proportionally to how often
SA/TA come up in bidding), and it's more conceptually coherent (the
"trump count" notion doesn't exist in SA).

Smoke-tested end to end (loading from `imit_bignet_ent02.pt`, 200
episodes, `aux_loss` already visibly decreasing, save/reload,
`eval_policy.py --architecture shared_aux`) -- no errors. **Not yet run
under real conditions.**

## First real run (`aux_coef=0.5`, seed30)

Same stable hyperparameters as `rl_experiments_v4/` (`lr=3e-5, epochs=4,
value_coef=0.2`, from `imit_bignet_ent02.pt`), `aux_coef=0.5` (a starting
value, not calibrated). Training healthy throughout (`clip_frac`
0.8-2.4%, normal range; `aux_loss` drops 0.35 -> ~0.20-0.22; `entropy`
stable ~0.20-0.24, no collapse).

**eval_avg(45000) = +15.84**, win_rate=53.3% -- clearly **below** the
shared-trunk-without-auxiliary-task reference (24.60, std=2.50, n=4) and
below REINFORCE alone (24.42, std=1.05, n=3). Even accounting for the
already-large inter-seed variance of the shared trunk (22.09 to 27.80
across the 4 seeds without aux), 15.84 is below the lowest observed so far
for this architecture family.

**Only one seed for now** -- no definitive conclusion yet (cf. the
`lr=1e-4` case in `rl_experiments_v4/README.md`, where a single clearly
worse seed was enough to drop that idea without digging further, the
alternative config already being confirmed over several seeds). Most
likely hypothesis: `aux_coef=0.5` is too high and the auxiliary loss
degrades the shared representation instead of improving it (the opposite
of the original reasoning -- a stronger gradient on the trunk is only
useful if its direction is compatible with what the actor needs).

## `aux_coef=0.1` (seed30): better, but still below the reference

Same protocol, only `aux_coef` changes (0.5 -> 0.1). Training healthy
again (`clip_frac` 0.9-2.7%, `aux_loss` drops 0.45 -> ~0.22-0.31).

**eval_avg(45000) = +19.64**, win_rate=54.1% -- better than `aux_coef=0.5`
(+15.84), confirming the hypothesis that a lower coefficient hurts the
shared representation less. Still below the shared-trunk-without-
auxiliary-task reference (24.60, std=2.50, n=4) and below REINFORCE alone
(24.42, n=3, std=1.05) -- the trend suggests an even lower `aux_coef`
might keep improving rather than having found an optimum.

| aux_coef | eval_avg(45000) |
|---|---|
| 0.5 | +15.84 |
| 0.1 | +19.64 |
| 0.02 | +24.19 |

## `aux_coef=0.02` (seed30): catches up to the reference, doesn't beat it

Monotonic trend confirmed: the lower `aux_coef` goes, the closer the
result gets to the shared-trunk reference WITHOUT the auxiliary task
(24.60, n=4). At 0.02, the gap (24.19 vs 24.60) is already within this
architecture's usual inter-seed noise (std=2.50). **No point on the curve
beats the reference** -- the evidence gathered so far points to a
neutral-to-slightly-negative effect of the auxiliary task at every
magnitude tested, the `aux_coef -> 0` limit simply converging toward "no
auxiliary task at all".

**Honest reading**: nothing in these 3 points supports the original
hypothesis (that the auxiliary gradient, by not having the critic's
shortcut, would improve the shared trunk more effectively than the critic
alone). On the contrary, even at low weight, it seems to slightly hinder
rather than help -- maybe because "predicting the opponents' trump count"
isn't a representation useful to the actor (which has to choose a card,
not estimate a quantity), unlike the AlphaStar intuition where the
auxiliary task targeted a more directly actionable signal.

## To do

- Decide whether to continue (e.g. test a weight even closer to 0 to
  confirm the convergence, or change the auxiliary target -- e.g. "aces
  remaining" rather than "trumps remaining") or consider this line
  inconclusive as it stands, like `lr=1e-4` in v4.
- If continuing: confirm the best config over several seeds before
  concluding anything, the same rigor as everywhere else -- but no point
  multiplying seeds on a config already dominated by the reference.
