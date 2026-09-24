# RL checkpoints - point 3/4 of remarques_rl.md

`imit.pt`: `CardNet` policy pretrained by supervised imitation of
`HeuristicPlayer` (`pretrain.py --games 5000 --epochs 20 --seed 0`), the
common starting point for every experiment below.

**Methodological note (corrected)**: up through the `exp_pool_350k`
comparison below, `eval_policy.py` reseeded `random` before every checkpoint
then let each game deal freshly, assuming that was enough to give every
checkpoint the same donnes. Wrong beyond the very first few games: as soon
as one checkpoint plays a different card from another,
`HeuristicPlayer._choose_defausse_suit()` (`HeuristicPlayer`'s only source
of randomness during play, outside bidding) consumes a different number of
`random` draws, and the whole global `random` state goes off on a different
trajectory for the rest of the loop -- verified empirically: hands diverged
as early as game 4 out of 100. Every `eval_avg`/`win_rate` table **before**
the `exp_pool_350k` section therefore comes from a comparison that wasn't
actually paired (noisier than what's written at the time), even though the
qualitative ranking is still corroborated by the corrected final comparison.
`eval_policy.py` now generates the N donnes once (independently of any
checkpoint, filtering out those that would trigger a silent internal
redeal, cf. game.py:139-143) then replays them identically via
`deal(hands=...)` for each checkpoint -- a real paired comparison.

## Entropy bonus decay (first series)

Two REINFORCE fine-tuning runs (`train.py --load imit.pt --episodes 100000
--eval-every 5000 --eval-games 500 --entropy-beta 0.02 --seed 5`,
counterfactual against heuristic active by default, cf. point 3), differing
only in the entropy bonus decay schedule (`--entropy-decay`):

- `exp_invsqrt/`: decay as `beta/sqrt(episode)`.
- `exp_inv/`: decay as `beta/episode`.

(`exp_none/`, the constant-entropy variant, was dropped: less conclusive
than the two above and redundant with `exp_lowlr_lowent` below, which
isolates the constant-entropy effect better via the ablation.)

Each directory contains a checkpoint every 5000 episodes
(`ckpt_ep5000.pt` ... `ckpt_ep100000.pt`), `final.pt` (identical to
`ckpt_ep100000.pt`), and `log.txt` (the full training output: raw reward,
counterfactual reward, greedy eval, win rate, effective beta).

Finding: neither run beat `imit.pt` alone on a robust evaluation (1500
games); both oscillate without clearly converging over 100k episodes. The
numeric detail is in each directory's `log.txt`. Cause identified later
(see below): `lr=1e-3` is too high for post-imitation fine-tuning,
independently of the entropy schedule.

## `exp_lowlr_lowent/`: lr and entropy reduced by a factor of 10 (kept)

Same starting point (`imit.pt`), but `lr=1e-4` and `entropy-beta=0.002`
(constant, `--entropy-decay none`), pushed to 100k episodes over 3
sequential runs (episodes 1-20000 seed=5, 20001-50000 seed=6, 50001-100000
seed=7, chained via `--load`/`--save` and `--episode-offset` for consistent
absolute numbering and different donnes on each resume). Only `final.pt`
(the weights at 100k) and `log.txt` (the full history of the 3 runs) are
kept; the intermediate checkpoints were deleted after analysis.

Ablation isolating each factor's effect (2x2, 20k episodes, seed=5,
compared on the same 1500 donnes via `eval_policy.py --games 1500` -- the
old, unpaired method, cf. the methodological note at the top of the file;
the lr-vs-entropy gap seen here is too large to be called into question,
but the precise values should be taken with more caution than what's
written):

|                    | beta=0.02 (high) | beta=0.002 (low) |
|--------------------|-----------------:|------------------:|
| **lr=1e-3 (high)** | -17.51 (exp_none, deleted) | -23.28 |
| **lr=1e-4 (low)**  | +0.74            | **+6.20**          |

(values: `eval_avg` over 1500 games vs `HeuristicPlayer`, same donnes for
every cell; heuristic-vs-heuristic mirror benchmark = +1.38.)

=> reducing `lr` carries essentially all of the gain; reducing
`entropy-beta` alone (lr stays at 1e-3) even makes the result worse.
Reducing both together (`exp_lowlr_lowent`) is the best combination found.

Progress over the 100k episodes (`eval_policy.py`, 1500 games, same
donnes): ep20000 = +6.20, ep50000 = +5.13, ep100000 (`final.pt`) = +5.79 —
a plateau reached within the first 20k episodes, no further net gain over
the following 80k episodes despite the budget spent.

## `exp_selfplay_frozen100k/`: self-play against a frozen copy (kept)

The plateau in `exp_lowlr_lowent` above suggested that `HeuristicPlayer` as
a fixed opponent stops "pushing" the policy once it already beats it on
average. Test: resume training from `exp_lowlr_lowent/final.pt` (seats
0/2), but against a **frozen copy** of the same checkpoint at seats 1/3
(`train.py --opponent`, greedy play, never updated) instead of against
`HeuristicPlayer`. Point 3's counterfactual now also follows the real
opponent (so the frozen copy at all 4 seats, instead of Heuristic -- cf.
`counterfactual_reward`'s docstring in `train.py`), to stay a reference
consistent with what's actually being played.

100k more episodes (absolute numbering 100001-200000, `--episode-offset
100000`, seed=8, same constant `lr=1e-4`/`entropy-beta=0.002` as
`exp_lowlr_lowent`). Only `final.pt` and `log.txt` are kept (same
conventions as above). Note: in `log.txt`, `eval_avg`/`win_rate` measure
progress **against the frozen copy** (close to 50%, expected since both
start from the same weights), not against `HeuristicPlayer`.

Result, measured on the real reference (`eval_policy.py --games 1500` vs
`HeuristicPlayer`, the old, unpaired method -- cf. the methodological note
at the top of the file):

| Policy | eval_avg(1500) | win_rate |
|---|---:|---:|
| `exp_selfplay_frozen100k/final.pt` | **+9.76** | **53.6%** |
| `exp_lowlr_lowent/final.pt` (starting point) | +5.79 | 50.7% |
| `heuristic` (mirror) | +1.38 | 49.7% |
| `imit.pt` | -3.54 | 49.7% |

=> self-play pushed the policy past the plateau observed against
`HeuristicPlayer` alone (+5.79 -> +9.76, almost double). Natural next step:
repeat the operation (self-play against a frozen copy of
`exp_selfplay_frozen100k/final.pt`) to see whether the gain reproduces or a
new plateau appears.

## `exp_selfplay_frozen150k/`: self-play round 2 (final kept checkpoint)

Same recipe, one more round: 50k more episodes (absolute numbering
200001-250000, seed=9) from `exp_selfplay_frozen100k/final.pt` (seats 0/2),
against a frozen copy of **the same checkpoint** at seats 1/3
(`exp_selfplay_frozen100k/final.pt` is still needed as this round's frozen
opponent and hasn't been deleted).

Unlike round 1, **no clear progress over this segment**: robust eval (1500
games vs `HeuristicPlayer`, same donnes, the old, unpaired method -- cf.
the methodological note at the top of the file) ep200000=+9.76,
ep210000=+8.49, ep220000=+6.33, ep230000=+15.34, ep240000=+4.23,
ep250000(`final.pt`)=+5.78 -- oscillation with no trend, the ep230000 spike
turned out to be mostly evaluation noise (drops back to +8.71 on a re-eval
at 10000 games, vs +6.51 for `final.pt` on the same sample -- a gap of ~2
points, under the standard error measured at this n (empirical reward
standard deviation per donne = 124 points, i.e. SE~1.25 at n=10000 -- this
SE calculation assumes a correctly paired sample, which wasn't yet the case
here: the real gap is probably even less significant than this figure).
The two checkpoints remain statistically indistinguishable; `final.pt` is
kept as the final checkpoint because it's the natural end of training, not
a choice cherry-picked on the eval.

Interpretation: one round of self-play against a frozen copy gives a real
gain (round 1), but continuing against the **same** frozen copy beyond that
no longer pushes the policy further -- probably because the fixed opponent
also becomes "beaten on average" and stops providing a useful gradient, as
observed with `HeuristicPlayer` in `exp_lowlr_lowent`. Next idea: refresh
the frozen opponent every round (a fictitious-self-play-style curriculum)
rather than repeating a round against the same snapshot.

**Lineage summary so far** (`eval_policy.py --games 1500`, vs
`HeuristicPlayer`, the old, unpaired method -- cf. the methodological note
at the top of the file): `imit.pt` -3.54 -> `exp_lowlr_lowent` +5.79 ->
`exp_selfplay_frozen100k` +9.76 -> `exp_selfplay_frozen150k` +5.78 (within
the previous round's noise, not a confirmed regression given the margin of
error). See the `exp_pool_350k` section below for the full summary with
the corrected, paired comparison.

## `exp_pool_350k/`: training against a pool of opponents (final kept checkpoint)

Self-play round 2 above didn't progress -- hypothesis: once the policy is
better on average than its single fixed opponent, that opponent stops
providing a useful gradient (the same mechanism as the `exp_lowlr_lowent`
plateau against `HeuristicPlayer` alone). Test: instead of a single fixed
opponent, `train.py --opponent` now samples a random opponent **on every
episode** from a pool (`heuristic,exp_selfplay_frozen100k
/final.pt,exp_selfplay_frozen150k/final.pt`); point 3's counterfactual
still follows whichever opponent was sampled that episode (cf.
`counterfactual_reward` in `train.py`), not a fixed opponent different from
what's actually being played. The eval logged during training stays vs
`HeuristicPlayer` by default (`evaluate()`), to stay comparable to previous
runs despite the pool.

100k episodes (absolute numbering 250001-350000, seed=10) from
`exp_selfplay_frozen150k/final.pt`. Only `final.pt` and `log.txt` are kept.

Result (`eval_policy.py --games 10000` vs `HeuristicPlayer`, **10000 donnes
generated once and replayed identically for every checkpoint** -- this
lineage's first comparison with a real pairing, cf. the methodological note
at the top of the file):

| Policy | eval_avg(10000) | win_rate |
|---|---:|---:|
| `imit.pt` | -1.52 | 49.2% |
| `heuristic` (mirror) | +3.49 | 50.8% |
| `exp_lowlr_lowent` | +5.83 | 51.5% |
| `exp_selfplay_frozen100k` | +8.07 | 52.1% |
| `exp_selfplay_frozen150k` | +8.68 | 52.9% |
| **`exp_pool_350k/final.pt`** | **+10.43** | **53.0%** |

=> the whole lineage still progresses monotonically once the comparison is
correctly paired, and the pool remains the best step so far. The
qualitative ranking obtained with the old (unpaired) method is confirmed;
only the absolute values differ (new donne sample, different filtering)
and aren't directly comparable to the tables in the earlier sections.
