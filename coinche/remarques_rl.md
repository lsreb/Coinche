# Notes on my understanding of rl_agent.py and how to improve it

## 1) The states
The 165-dim state space fed into the neural network we want to train seems poorly chosen to me. It probably doesn't really account for the symmetry between each suit, and tries to learn from so much data that it doesn't really reflect what a real player perceives. Should the state space be changed to only capture the natural signals (suit lengths, bids, play order, finesses, etc.)? Should the input states be restricted, or should the network be left to handle that itself?

## 2) The neural network
The neural network could be the place to account for the signals I mention in 1) if the states themselves don't capture them. Should its structure change? If so, how?

## 3) Reinforcement
The reward used for reinforcement is the "reward" quantity after a game. Given how deterministic and rule-bound Coinche's play style is, maybe this reward should actually be compared against the reward we would have gotten had Heuristic played instead of RL, and that reward difference used as the real reward. Or is this concept already more or less captured by the baseline? Or would it be a good way to initialize the baseline, or is it something to do on top of the baseline?


## 4) The initial policy
Rather than starting from a random card choice initially, which is the case if I understand correctly, wouldn't it be better to start from a choice close to the heuristic + some randomization around it, in hopes of starting from a more reasonable choice?

## 5) PPO / REINFORCE comparison: hyperparameters reused as-is without revalidating them
For `train_ppo.py`, I reused REINFORCE's `lr=1e-4` (the lr/entropy ablation from `rl_experiments/README.md`) as a starting point, to compare under equal protocol. But a `--lr` sweep on the PPO side shows `3e-5` is clearly better than `1e-4` (see `rl_experiments_v2/README.md`) — and, more importantly, that it isn't monotonic (`8e-4` destroys the policy, `1e-5` underlearns). Now, this `lr=1e-4` for REINFORCE itself comes from an ablation done *before* the scoring rule-5 fix, which raised the reward's empirical standard deviation from ~124 to ~228 points/donne (a much noisier signal now). Nothing guarantees this setting is still optimal for REINFORCE under the new signal. Before concluding anything definitive about PPO vs REINFORCE, shouldn't the same lr/entropy ablation be redone on REINFORCE (`train.py`) under the post-rule-5 signal, rather than comparing a freshly tuned PPO to a REINFORCE whose settings were never re-checked?

## 6) Plan: bigger MLP + shared-trunk centralized critic + auxiliary tasks (2026-07-26 — DONE AND SETTLED, cf. the 2026-07-30 update at the end of the file)

Context: in `rl_experiments_v3`, the growing pool + PFSP (`run_growing_pool.py`/`run_growing_pool_ppo.py`) is the only structural idea that has held up under full-sample testing (n=3 REINFORCE, n=2 PPO in progress); every critic-side idea (partial, isolated centralized, pretraining, enriched target, present/absent) has failed to show a measurable effect — probably because the critic (`ValueNet`) is a network entirely independent of the actor (`CardNet`), so even a better critic-side info/representation can never reach the actor.

Chosen plan, to be tested in 2 isolated steps (not all at once, to avoid confounding the variables):

1. **Capacity alone, on REINFORCE first**: `CardNet` goes from 1 to 3 hidden layers (currently `fc1` (in→hidden) + `fc2` (hidden→32), a single hidden layer), everything else identical (networks still separate, plain training against heuristic). To compare against the ~15.1-15.2 already established over several seeds (`epochs=8` PPO / REINFORCE `lr=3e-5`). Hypothesis: 128 units on a single layer may be a real capacity ceiling (imitation saturates at ~92-93% accuracy, not 99%).

2. **Shared trunk + centralized critic + auxiliary tasks**, tested separately from (1): a common trunk (1-2 hidden layers) fed by `encode_state` (167-dim, what the actor also sees), shared between policy and critic — unlike the current architecture where `policy_net`/`value_net` are independent. The actor then branches off on its own (1 private hidden layer + decision head). The critic additionally receives the centralized info (the 3 others' hands, `encode_full_state`/`_other_hands_vec`, already implemented) through a small separate branch merged after the shared trunk, plus 1-2 additional private hidden layers, and can incorporate auxiliary-task heads (e.g. the number of trumps/aces remaining for the 3 other players — inspired by AlphaStar's centralized critic, cf. the 2026-07-26 discussion). With the trunk shared, the value gradient (and the auxiliary tasks') also shapes the representation the actor uses — unlike the centralized critic already tested (isolated, so with no possible effect on the actor by construction).
   - Value head target: **the counterfactual-adjusted return** (as today, `reward - counterfactual_reward(...)`), NOT the raw return -- already verified empirically (`pretrain_value.py`) that regressing on the raw return gives a confident but biased critic (an attack/defense gap of ~190 points on the raw return, only ~17 points once the counterfactual is subtracted).
   - Risk to watch (already hit once, cf. point 3 and this lineage's PPO history): `train_ppo.py`'s original shared-trunk architecture (`ActorCriticNet`) was abandoned after a bug where `value_loss` (scale of hundreds of points) dominated `policy_loss` (scale O(1)) and corrupted the policy's gradient -- fixed once via scaling (`return_scale = returns.std()`), then the networks were separated out of caution. Going back to a shared trunk, with potentially several auxiliary losses on top, multiplies the relative coefficients to calibrate (`value_coef`, one `aux_coef` per task) and the risk that one term dominates and pollutes the shared representation.

Implementation: don't modify `CardNet`/`ValueNet`/`PPOPolicy`/`NeuralPolicy` in place (keep the already-validated configs reproducible as-is) -- add new classes alongside them (new names, same file or a new module) for these architectures, reusing as-is everything that's already solid and orthogonal to this change (the game engine, `encode_state`/`encode_full_state`, the PFSP pool, the counterfactual, the eval scripts `eval_policy.py`/`eval_matchup.py`, the `run_growing_pool*.py` orchestrators).

## 7) 2026-07-30 update: what point 6's plan produced, and where things really stand

Both steps of the plan were implemented (`rl_experiments_v4/`, `rl_experiments_v5/`), and the verdict is clear:

- **Step 1 (capacity alone, `CardNetBig`)**: a massive gain, but not for the capacity reason itself -- diagnosed that imitation run to full convergence made the policy far too confident/inflexible (entropy ~0.085 vs ~0.215 for `CardNet`). Fixed by stopping imitation early, at a target entropy (~0.20) rather than at convergence. This one change alone (`imit_bignet_ent02.pt`) gave the biggest confirmed gain of this project's entire RL experimentation (+9.27 points vs the `CardNet` plateau, n=3, highly significant).
- **Step 2 (shared trunk + centralized critic)**: implemented (`SharedTrunkActorCritic`), works (the critic's gradient does reach the actor, unlike the 4 previous independent-critic attempts) but **only matches plain REINFORCE, without beating it** (24.60 vs 24.42, n=4/3). `--no-critic-baseline` re-tested specifically on this architecture: still no net gain from the critic (the 7th null result of this kind on this project).
- **Auxiliary tasks** (`SharedTrunkActorCriticAux`, `rl_experiments_v5/`): implemented with the right care (the auxiliary head isolated from the critic's centralized-info branch, so it can't "cheat"). A sweep of `aux_coef` (0.5/0.1/0.02): simply converges toward "no auxiliary task" as the coefficient goes down, **no gain**.
- **Trunk/head rebalancing** (`SharedTrunkActorCriticDeep`, 3-layer trunk / 1-layer actor head instead of 2+2): tested at n=2, **neutral**, within the original trunk's distribution.
- **Conclusion on the whole "critic/shared-trunk/auxiliary-tasks" line**: mechanically it works (the gradient does reach the actor), but empirically **nothing beats plain REINFORCE from the same starting point**, on any variant tested. I now consider this line fully explored, not a priority to continue unless a qualitatively different new idea comes up.

**The real lever confirmed this session is the growing pool (self-play + PFSP)**, not the critic. Confirmed by round-robin (not just `eval_avg` against heuristic alone, which has proven misleading several times -- specialization drift that looks like regression but isn't):
- Pool + `CardNetBig`/REINFORCE (from `imit_bignet_ent02.pt`): a real, confirmed gain, pushed up to 750k episodes, near-linear progress (+~5 corrected-strength points every 150k) up to ~600k, then a clear slowdown (+2.08 at the last step) -- diminishing returns, not yet a real ceiling.
- Pool + shared trunk/PPO: also a real gain (confirmed by round-robin after an initially misleading reading via `eval_avg` alone).
- Direct head-to-head between the session's two best results (pool bignet+REINFORCE vs pool shared-trunk+PPO), n=100000 (the finest precision of the session): **statistically tied**. The base architecture choice no longer seems to matter once the pool is added -- it's the pool doing the work.

**Rules bug found and fixed along the way (2026-07-29)**: `legal_moves()` didn't enforce the overtrump obligation in TA (wrongly treated like SA). Impact assessed: negligible in practice (the heuristic already complied with the correct rule through its own play logic, independently of what the engine permitted) -- confirmed empirically by replaying the same round-robins under the fixed engine, near-identical gaps. No retrain needed.

**Ongoing diagnosis of the growing pool's slowdown (2026-07-30)**: at a fixed segment budget (50k episodes), the bigger the pool grows, the less practice each opponent (especially the most recent, most relevant one) gets -- measured: the share of games against the latest addition dropped from ~15% to ~8% between 300k and 750k, at the same time as the per-step gain collapsed. Built three levers in `run_growing_pool.py` to compensate:
- `--pool-keep-every`: regular decimation of the pool by segment index (keeps 1 out of N + the most recent).
- `--pool-max-size`: adaptive weakness-based pruning -- drops the opponent with the highest win rate (the least tough) when the pool exceeds a cap, from the PFSP summary already logged at the end of the segment (no memory needs to persist between segments, each segment restarts from scratch anyway).
- `--pool-exclude`: manual, permanent exclusion of specific segments by episode number.
- `--pfsp-explore-eps`: an exploration floor (softmax + uniform sampling mix) to offset the risk that an aggressive PFSP temperature (lowered from 0.1 to 0.05 to bias more strongly toward tough opponents) completely wipes out practice against an easier opponent.

All of this was just tested (in isolation, smoke tests + a real extension to 900k in progress with the new settings) -- not enough hindsight yet to know whether it really reignites progress or whether 750k-900k is the pool's real ceiling at this scale.

**Never-tried idea, identified but not explored**: the growing pool in PPO with `CardNetBig` alone (no shared trunk) -- the only missing combination in the {REINFORCE, PPO} × {CardNetBig, shared trunk} grid.
