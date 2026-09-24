"""Trains an actor-critic card-play policy via PPO (Schulman et al. 2017),
an alternative to REINFORCE (train.py) on exactly the same task: same
state/action (`coinche.rl_agent.encode_state`, 32 canonical slots), same RL
team at seats 0/2 sharing weights, same opponent(s) at seats 1/3
(--opponent, uniform or PFSP pool), same point-3 counterfactual
(`train.counterfactual_reward`). Deliberately a separate file from train.py
so PPO can be developed/tested without risking an already-running REINFORCE
experiment (the code is re-read at the start of each process, not "live" --
but it's still worth avoiding any ambiguity about which code produced which
checkpoint).

Key difference from REINFORCE: instead of a single scalar reward per donne
applied to the sum of the trajectory's log-probs with a global EMA baseline
(`NeuralPolicy.update`), PPO:
  1) accumulates a batch of several donnes (`--batch-episodes`) before
     updating;
  2) uses a critic (`ValueNet`) that learns to predict the expected return
     FROM EACH STATE of the trajectory -- a finer baseline than a global
     moving average, advantage = the donne's return minus this predicted
     value;
  3) runs several passes (`--epochs`) of minibatch descent over this batch,
     with a clipped probability ratio (`--clip-eps`) between the current
     policy and the one that collected the trajectory, which allows several
     updates per collected episode without diverging (unlike REINFORCE,
     which does one strictly on-policy update per episode).
The return used for every timestep of a donne is the same scalar for the
whole donne (terminal reward, cf. the point-3 counterfactual): there's no
intermediate per-trick reward defined in this game.

`ValueNet` has its own trunk, INDEPENDENT of the policy's (`CardNet`, shared
with REINFORCE/imit.pt): a shared trunk inherited from imit.pt (never
trained for value) left the critic nearly unable to separate attack/defense
even after 50k episodes (cf. remarques_rl.md) -- pretraining `ValueNet` by
supervised regression on the final return (`pretrain_value.py`, the value
counterpart of `pretrain.py`) before PPO fine-tuning is therefore
recommended.

`ValueNet` is also a CENTRALIZED critic (`encode_full_state`, cf.
rl_agent.py): unlike the policy, which only sees what a real player would
see, the critic additionally receives the current hands of the 3 other
seats -- privileged info, known only because we simulate the whole donne
during training, never available to the actor in real play. Motivated by
the low ceiling measured for a partial-info critic (R2~0.04-0.06 even when
well trained, cf. remarques_rl.md): this game's variance is dominated by
hidden information that no critic conditioned on a partial state can guess,
but nothing prevents raising this ceiling for the critic since it's already
a network independent of the policy.

Usage:
    python train_ppo.py --episodes 5000 --eval-every 200
    python train_ppo.py --episodes 2000 --load weights.pt --save weights.pt
    python train_ppo.py --load imit.pt --load-value value_imit.pt --episodes 50000
    python train_ppo.py --load weights.pt --opponent heuristic,100k.pt,250k.pt --pfsp --episodes 100000
"""
import argparse
import os
import random
import time

import torch.nn as nn
import torch.nn.functional as F

from coinche.rl_agent import (
    STATE_DIM, FULL_STATE_DIM, CardNet, CardNetBig, SharedTrunkActorCritic,
    SharedTrunkActorCriticAux, SharedTrunkActorCriticDeep, encode_state, encode_full_state,
    other_trump_counts, _canonical_slots, _slot_index, torch,
)
from train import (
    build_opponent_pool, PFSPSampler, run_episode, counterfactual_reward,
    evaluate, entropy_beta_for_episode,
)


if torch is not None:
    class ValueNet(nn.Module):
        """Trunk INDEPENDENT of the policy (no sharing with CardNet), 1
        hidden layer -> 1 scalar. Before the split, value_head shared the
        policy's `fc1` trunk (inherited from imit.pt, never trained for
        value): a diagnostic (remarques_rl.md) showed that after 50k PPO
        episodes the critic barely separated attack/defense (is_attacker,
        even though it's a direct input feature) even though the real
        return gap between the two is huge -- value_loss's gradient on this
        shared trunk was overwhelmed by policy_loss's. A dedicated trunk,
        independently pretrainable (pretrain_value.py), avoids this
        competition.

        Takes FULL_STATE_DIM (encode_full_state) as input, not STATE_DIM:
        a CENTRALIZED critic, which also sees the current hands of the 3
        other seats (privileged info, known only during training
        simulation, never to the actor in real play -- cf. remarques_rl.md,
        motivated by the low ceiling of a partial-info critic: R2~0.04-0.06
        even when well trained, for lack of being able to guess the
        opponents' hands). Nothing prevents this
        partial-info-for-the-actor / full-info-for-the-critic asymmetry
        since the two are independent networks."""
        def __init__(self, in_dim=FULL_STATE_DIM, hidden=128):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            return self.value_head(h).squeeze(-1)


    class PPOPolicy:
        """Interface compatible with RLPlayer/run_episode/evaluate
        (choose_card, record, save, load) so train.py's functions can be
        reused as-is. `record=True` (default) samples and stores (the
        state visible to the actor, the centralized state for the critic,
        log-prob, value, legal-move mask) into `self._traj`;
        `record=False` (evaluation) plays greedily (argmax) without
        storing anything."""
        def __init__(self, device='cpu', lr=1e-3, clip_eps=0.2, value_coef=0.5,
                     entropy_coef=0.01, epochs=4, minibatch_size=64, no_critic_baseline=False,
                     ablate_points=False, policy_net_cls=CardNet):
            self.device = device
            self.policy_net = policy_net_cls().to(device)
            self.value_net = ValueNet().to(device)
            self.optimizer = torch.optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
            self.clip_eps = clip_eps
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.epochs = epochs
            self.minibatch_size = minibatch_size
            # If True: advantage = return (centered/normalized on the current BATCH's mean,
            # cf. update_batch), value_net is neither used nor trained -- tests whether a
            # per-state critic adds anything over a simple REINFORCE-style scalar.
            self.no_critic_baseline = no_critic_baseline
            # Ablation study: forces encode_state's `_points_so_far` block to zero (on
            # both sides, actor and critic) -- cf. rl_experiments_v3/README.md.
            self.ablate_points = ablate_points
            self.record = True
            self._traj = []       # (state, full_state, action_idx, log_prob, value, mask) of the current donne
            self._episodes = []   # [(traj, return)] accumulated since the last update_batch()

        def choose_card(self, player, legal, leader, trick, trump):
            x = torch.tensor(encode_state(player, trick, trump, ablate_points=self.ablate_points),
                              dtype=torch.float32, device=self.device)
            logits = self.policy_net(x)
            suits, orders = _canonical_slots(trump)
            mask = torch.full((32,), float('-inf'), device=self.device)
            legal_idx = [_slot_index(c.suit, c.rank, suits, orders) for c in legal]
            mask[legal_idx] = 0.0
            masked_logits = logits + mask

            if self.record:
                # Centralized state (opponents' hands included) only for the critic,
                # never for the policy above -- cf. ValueNet's docstring.
                x_full = torch.tensor(encode_full_state(player, trick, trump, ablate_points=self.ablate_points),
                                       dtype=torch.float32, device=self.device)
                value = self.value_net(x_full)
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                self._traj.append((x.detach(), x_full.detach(), int(action_idx.item()),
                                    dist.log_prob(action_idx).detach(), value.detach(),
                                    mask.detach()))
            else:
                action_idx = torch.argmax(masked_logits)

            slot, rank_pos = divmod(int(action_idx.item()), 8)
            chosen_suit, chosen_rank = suits[slot], orders[slot][rank_pos]
            return next(c for c in legal if c.suit == chosen_suit and c.rank == chosen_rank)

        def end_episode(self, reward):
            """Call once the donne is finished (instead of `.update()` for
            REINFORCE): archives the trajectory with its return (the
            counterfactual reward, as in REINFORCE) for the next
            `update_batch()`. Triggers no immediate update."""
            if not self._traj:
                return
            self._episodes.append((self._traj, reward))
            self._traj = []

        def update_batch(self):
            """PPO update over all the donnes accumulated since the last
            call (cf. `end_episode`): advantage = the donne's return minus
            the predicted value at that state (same return for all
            timesteps of a donne, terminal reward), normalized, then
            `self.epochs` minibatch passes over the clipped objective +
            value loss (MSE) + entropy bonus. Clears the batch afterward.
            If `self.no_critic_baseline`: no predicted-value subtraction --
            the normalized advantage below then becomes a centering on the
            current BATCH's mean (REINFORCE-style, but a per-batch scalar
            rather than a global EMA), `value_net` is neither called nor
            trained (`value_loss` returned as 0).
            Returns the average (policy_loss, value_loss, entropy,
            clip_frac) (clip_frac: fraction of samples where |ratio-1| >
            clip_eps, a standard PPO diagnostic -- measurement only,
            doesn't change behavior), or None if nothing was accumulated."""
            if not self._episodes:
                return None
            states, full_states, actions, old_log_probs, old_values, masks, returns = [], [], [], [], [], [], []
            for traj, reward in self._episodes:
                for x, x_full, action_idx, old_lp, old_v, mask in traj:
                    states.append(x)
                    full_states.append(x_full)
                    actions.append(action_idx)
                    old_log_probs.append(old_lp)
                    old_values.append(old_v)
                    masks.append(mask)
                    returns.append(reward)
            self._episodes = []

            states = torch.stack(states)
            full_states = torch.stack(full_states)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            old_log_probs = torch.stack(old_log_probs)
            old_values = torch.stack(old_values)
            masks = torch.stack(masks)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages = returns if self.no_critic_baseline else returns - old_values
            # Scale of the raw returns (~hundreds of points, cf. rl_experiments_v2/README.md):
            # without this, value_loss (MSE on these returns) dominates policy_loss (on the
            # normalized advantage, O(1)) by several orders of magnitude -- policy_net and
            # value_net are now independent trunks (no more risk of "polluting" the policy's
            # features), but `--value-coef` would remain without an interpretable effect
            # without this common scaling.
            return_scale = returns.std() + 1e-6
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            n = states.shape[0]
            order = list(range(n))
            total_policy_loss = total_value_loss = total_entropy = total_clip_frac = 0.0
            n_updates = 0
            for _ in range(self.epochs):
                random.shuffle(order)
                for start in range(0, n, self.minibatch_size):
                    mb = torch.tensor(order[start:start + self.minibatch_size], dtype=torch.long, device=self.device)
                    logits = self.policy_net(states[mb])
                    # Reapply the legal-move mask (choose_card): without it, new_log_probs
                    # would be computed under a distribution over all 32 slots (including
                    # illegal moves), different from the one old_log_probs was sampled under
                    # -- the PPO ratio below would then lose its "how much has the policy
                    # moved" meaning.
                    dist = torch.distributions.Categorical(logits=logits + masks[mb])
                    new_log_probs = dist.log_prob(actions[mb])
                    ratio = torch.exp(new_log_probs - old_log_probs[mb])
                    adv = advantages[mb]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy = dist.entropy().mean()
                    if self.no_critic_baseline:
                        value_loss = torch.tensor(0.0)
                        loss = policy_loss - self.entropy_coef * entropy
                    else:
                        values = self.value_net(full_states[mb])
                        value_loss = F.mse_loss(values / return_scale, returns[mb] / return_scale)
                        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    # Measurement only (doesn't affect loss/gradient): how often the clip
                    # is actually in play (|ratio-1| > clip_eps), standard PPO convention.
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_clip_frac += clip_frac.item()
                    n_updates += 1

            return (total_policy_loss / n_updates, total_value_loss / n_updates,
                    total_entropy / n_updates, total_clip_frac / n_updates)

        def save(self, path):
            """Native PPO checkpoint: both independent networks in a
            single file (a dict with 2 keys). For a full resume, pass this
            same path to both --load and --load-value (each reads its own
            share). Does NOT save self.optimizer's state (Adam's m/v
            moments): resuming (--load/--load-value then continuing
            training in a new process) restarts with a "cold" Adam on
            already-trained weights, not a truly continuous history --
            the same limitation already present for NeuralPolicy/REINFORCE
            (train.py) between rl_experiments_v2/exp_growing_pool's
            segments. Estimated impact is small (the moments restabilize
            within a few hundred steps, a negligible fraction of a segment
            of tens of thousands of episodes) but real at every
            --episode-offset boundary."""
            torch.save({'policy': self.policy_net.state_dict(),
                        'value': self.value_net.state_dict()}, path)

        def load(self, path):
            """Loads the policy from: a native PPO checkpoint (dict,
            'policy' key), a REINFORCE checkpoint (CardNet: fc1/fc2, cf.
            rl_agent.py -- imit.pt or rl_experiments(_v2)/ segments, same
            shape as policy_net so no remapping needed), or an older
            shared-trunk PPO checkpoint (fc1/policy_head/value_head, before
            the networks were split -- value_head is then ignored)."""
            obj = torch.load(path, map_location=self.device)
            if 'policy' in obj:
                self.policy_net.load_state_dict(obj['policy'])
            elif 'fc2.weight' in obj:
                self.policy_net.load_state_dict(obj)
            elif 'policy_head.weight' in obj:
                state_dict = dict(obj)
                state_dict['fc2.weight'] = state_dict.pop('policy_head.weight')
                state_dict['fc2.bias'] = state_dict.pop('policy_head.bias')
                state_dict.pop('value_head.weight', None)
                state_dict.pop('value_head.bias', None)
                self.policy_net.load_state_dict(state_dict)
            else:
                raise ValueError(f"Unrecognized checkpoint format for the policy: {path}")

        def load_value(self, path):
            """Loads value_net from: a native PPO checkpoint (dict,
            'value' key), or a value checkpoint pretrained by
            pretrain_value.py (same shape as ValueNet, no remapping needed)."""
            obj = torch.load(path, map_location=self.device)
            self.value_net.load_state_dict(obj['value'] if 'value' in obj else obj)

    class SharedTrunkPPOPolicy:
        """Variant of PPOPolicy for step 2 of the plan (remarques_rl.md
        point 6, 2026-07-27 discussion): actor and critic share a single
        network (SharedTrunkActorCritic) instead of two independent trunks
        (CardNet + ValueNet). Same interface as PPOPolicy (choose_card,
        end_episode, update_batch, save, load) to stay compatible with
        RLPlayer/run_episode/evaluate/eval_policy.py without modifying them.

        Unlike PPOPolicy, there is now only a single network/optimizer
        (the shared trunk receives the gradient of both policy_loss and
        value_loss) -- exactly the mechanism that was missing from
        previous attempts on the critic (all with an independent ValueNet,
        so with no possible effect on the actor by construction)."""
        def __init__(self, device='cpu', lr=1e-3, clip_eps=0.2, value_coef=0.5,
                     entropy_coef=0.01, epochs=4, minibatch_size=64, no_critic_baseline=False,
                     net_cls=SharedTrunkActorCritic):
            self.device = device
            self.net = net_cls().to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.clip_eps = clip_eps
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.epochs = epochs
            self.minibatch_size = minibatch_size
            # Cf. PPOPolicy.no_critic_baseline (same semantics): advantage = return
            # centered/normalized on the batch, the critic head is neither called nor
            # trained in update_batch -- tests whether the critic's baseline adds
            # anything HERE (never tested on the shared trunk before, unlike the old
            # independent critic, ablated 3 times with no measurable effect).
            self.no_critic_baseline = no_critic_baseline
            self.record = True
            self._traj = []
            self._episodes = []

        def choose_card(self, player, legal, leader, trick, trump):
            x = torch.tensor(encode_state(player, trick, trump), dtype=torch.float32, device=self.device)
            logits = self.net.forward_actor(x)
            suits, orders = _canonical_slots(trump)
            mask = torch.full((32,), float('-inf'), device=self.device)
            legal_idx = [_slot_index(c.suit, c.rank, suits, orders) for c in legal]
            mask[legal_idx] = 0.0
            masked_logits = logits + mask

            if self.record:
                x_full = torch.tensor(encode_full_state(player, trick, trump),
                                       dtype=torch.float32, device=self.device)
                value = self.net.forward_critic(x_full)
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                self._traj.append((x.detach(), x_full.detach(), int(action_idx.item()),
                                    dist.log_prob(action_idx).detach(), value.detach(),
                                    mask.detach()))
            else:
                action_idx = torch.argmax(masked_logits)

            slot, rank_pos = divmod(int(action_idx.item()), 8)
            chosen_suit, chosen_rank = suits[slot], orders[slot][rank_pos]
            return next(c for c in legal if c.suit == chosen_suit and c.rank == chosen_rank)

        def end_episode(self, reward):
            if not self._traj:
                return
            self._episodes.append((self._traj, reward))
            self._traj = []

        def update_batch(self):
            """Identical to PPOPolicy.update_batch, except policy_net/value_net
            are replaced by the same shared network's two
            forward_actor/forward_critic methods -- a single optimizer, a
            single .backward() per minibatch (value_loss's gradient also
            flows through the shared trunk)."""
            if not self._episodes:
                return None
            states, full_states, actions, old_log_probs, old_values, masks, returns = [], [], [], [], [], [], []
            for traj, reward in self._episodes:
                for x, x_full, action_idx, old_lp, old_v, mask in traj:
                    states.append(x)
                    full_states.append(x_full)
                    actions.append(action_idx)
                    old_log_probs.append(old_lp)
                    old_values.append(old_v)
                    masks.append(mask)
                    returns.append(reward)
            self._episodes = []

            states = torch.stack(states)
            full_states = torch.stack(full_states)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            old_log_probs = torch.stack(old_log_probs)
            old_values = torch.stack(old_values)
            masks = torch.stack(masks)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages = returns if self.no_critic_baseline else returns - old_values
            return_scale = returns.std() + 1e-6
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            n = states.shape[0]
            order = list(range(n))
            total_policy_loss = total_value_loss = total_entropy = total_clip_frac = 0.0
            n_updates = 0
            for _ in range(self.epochs):
                random.shuffle(order)
                for start in range(0, n, self.minibatch_size):
                    mb = torch.tensor(order[start:start + self.minibatch_size], dtype=torch.long, device=self.device)
                    logits = self.net.forward_actor(states[mb])
                    dist = torch.distributions.Categorical(logits=logits + masks[mb])
                    new_log_probs = dist.log_prob(actions[mb])
                    ratio = torch.exp(new_log_probs - old_log_probs[mb])
                    adv = advantages[mb]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy = dist.entropy().mean()
                    if self.no_critic_baseline:
                        value_loss = torch.tensor(0.0)
                        loss = policy_loss - self.entropy_coef * entropy
                    else:
                        values = self.net.forward_critic(full_states[mb])
                        value_loss = F.mse_loss(values / return_scale, returns[mb] / return_scale)
                        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_clip_frac += clip_frac.item()
                    n_updates += 1

            return (total_policy_loss / n_updates, total_value_loss / n_updates,
                    total_entropy / n_updates, total_clip_frac / n_updates)

        def save(self, path):
            """Native checkpoint: a single network, unlike PPOPolicy.save()'s
            2-key dict (no more separate policy_net/value_net)."""
            torch.save(self.net.state_dict(), path)

        def load(self, path):
            """Loads either a native checkpoint of this architecture (at
            least one '*_actor.weight' key -- SharedTrunkActorCritic has
            both fc3_actor AND fc4_actor, SharedTrunkActorCriticDeep only
            fc4_actor, hence this generic test rather than a fixed key
            name), or a raw CardNetBig checkpoint (e.g.
            imit_bignet_ent02.pt, no '*_actor' key) via
            net_cls.load_actor_from_cardnetbig -- the critic part then
            starts randomly initialized."""
            obj = torch.load(path, map_location=self.device)
            if any(k.endswith('_actor.weight') for k in obj):
                self.net.load_state_dict(obj)
            elif 'fc3.weight' in obj:
                self.net.load_actor_from_cardnetbig(path, map_location=self.device)
            else:
                raise ValueError(f"Unrecognized checkpoint format for SharedTrunkPPOPolicy: {path}")

    class SharedTrunkAuxPPOPolicy(SharedTrunkPPOPolicy):
        """SharedTrunkPPOPolicy + an auxiliary task (rl_experiments_v5,
        2026-07-27 discussion): SharedTrunkActorCriticAux additionally
        predicts the number of trumps remaining for the 3 other seats,
        from the shared trunk ALONE (see SharedTrunkActorCriticAux's
        docstring in rl_agent.py for the full reasoning -- deliberately
        NOT from the critic's centralized info branch, to prevent the
        network from "cheating" by reading the answer directly out of that
        branch rather than having to encode it in the trunk shared with
        the actor).

        New hyperparameter `aux_coef` (this loss's weight in the total
        loss) -- the auxiliary loss is normalized by its own standard
        deviation on the current batch (same logic as `return_scale` for
        `value_loss`), without which `aux_coef` wouldn't have an
        interpretable meaning (the trump-count scale, 0 to 8, is very
        different from that of the normalized advantage or the entropy).

        In SA/TA, "canonical slot 0" (cf. other_trump_counts) doesn't
        correspond to a real trump (none in SA, all 4 suits tied in TA) --
        these decisions are excluded from the auxiliary loss (`aux_valid`
        mask, 2026-07-27 discussion), without affecting policy_loss/
        value_loss, which keep training normally on them.

        `update_batch` returns a 5-tuple (policy_loss, value_loss,
        aux_loss, entropy, clip_frac) instead of the parent class's
        4-tuple -- handled explicitly by train_ppo.py's `main()`."""
        def __init__(self, device='cpu', lr=1e-3, clip_eps=0.2, value_coef=0.5,
                     aux_coef=0.5, entropy_coef=0.01, epochs=4, minibatch_size=64):
            self.device = device
            self.net = SharedTrunkActorCriticAux().to(device)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
            self.clip_eps = clip_eps
            self.value_coef = value_coef
            self.aux_coef = aux_coef
            self.entropy_coef = entropy_coef
            self.epochs = epochs
            self.minibatch_size = minibatch_size
            self.record = True
            self._traj = []
            self._episodes = []

        def choose_card(self, player, legal, leader, trick, trump):
            x = torch.tensor(encode_state(player, trick, trump), dtype=torch.float32, device=self.device)
            logits = self.net.forward_actor(x)
            suits, orders = _canonical_slots(trump)
            mask = torch.full((32,), float('-inf'), device=self.device)
            legal_idx = [_slot_index(c.suit, c.rank, suits, orders) for c in legal]
            mask[legal_idx] = 0.0
            masked_logits = logits + mask

            if self.record:
                x_full = torch.tensor(encode_full_state(player, trick, trump),
                                       dtype=torch.float32, device=self.device)
                value = self.net.forward_critic(x_full)
                # Auxiliary task target: privileged info (complete hands),
                # computed here (player/trump context available), not at
                # update_batch time (states alone, no more game context).
                aux_target = torch.tensor(other_trump_counts(player, trump),
                                           dtype=torch.float32, device=self.device)
                # In SA/TA, "canonical slot 0" doesn't correspond to a real trump
                # (none in SA, all 4 suits tied in TA, cf. the 2026-07-27
                # discussion) -- the target then has no strategic meaning, so
                # these decisions are excluded from the auxiliary loss (but not
                # from policy_loss/value_loss, which keep training normally on
                # SA/TA just as on suit contracts).
                aux_valid = trump not in ('SA', 'TA')
                dist = torch.distributions.Categorical(logits=masked_logits)
                action_idx = dist.sample()
                self._traj.append((x.detach(), x_full.detach(), aux_target, aux_valid,
                                    int(action_idx.item()),
                                    dist.log_prob(action_idx).detach(), value.detach(),
                                    mask.detach()))
            else:
                action_idx = torch.argmax(masked_logits)

            slot, rank_pos = divmod(int(action_idx.item()), 8)
            chosen_suit, chosen_rank = suits[slot], orders[slot][rank_pos]
            return next(c for c in legal if c.suit == chosen_suit and c.rank == chosen_rank)

        def update_batch(self):
            if not self._episodes:
                return None
            states, full_states, aux_targets, aux_valids = [], [], [], []
            actions, old_log_probs, old_values, masks, returns = [], [], [], [], []
            for traj, reward in self._episodes:
                for x, x_full, aux_target, aux_valid, action_idx, old_lp, old_v, mask in traj:
                    states.append(x)
                    full_states.append(x_full)
                    aux_targets.append(aux_target)
                    aux_valids.append(aux_valid)
                    actions.append(action_idx)
                    old_log_probs.append(old_lp)
                    old_values.append(old_v)
                    masks.append(mask)
                    returns.append(reward)
            self._episodes = []

            states = torch.stack(states)
            full_states = torch.stack(full_states)
            aux_targets = torch.stack(aux_targets)
            aux_valids = torch.tensor(aux_valids, dtype=torch.bool, device=self.device)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            old_log_probs = torch.stack(old_log_probs)
            old_values = torch.stack(old_values)
            masks = torch.stack(masks)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages = returns - old_values
            return_scale = returns.std() + 1e-6
            # Normalization computed only over the valid decisions (suit
            # contract) -- excludes the meaningless SA/TA targets, cf. choose_card.
            aux_scale = (aux_targets[aux_valids].std() + 1e-6) if aux_valids.any() else torch.tensor(1.0)
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            n = states.shape[0]
            order = list(range(n))
            total_policy_loss = total_value_loss = total_aux_loss = total_entropy = total_clip_frac = 0.0
            n_updates = 0
            for _ in range(self.epochs):
                random.shuffle(order)
                for start in range(0, n, self.minibatch_size):
                    mb = torch.tensor(order[start:start + self.minibatch_size], dtype=torch.long, device=self.device)
                    logits = self.net.forward_actor(states[mb])
                    dist = torch.distributions.Categorical(logits=logits + masks[mb])
                    new_log_probs = dist.log_prob(actions[mb])
                    ratio = torch.exp(new_log_probs - old_log_probs[mb])
                    adv = advantages[mb]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy = dist.entropy().mean()
                    values = self.net.forward_critic(full_states[mb])
                    value_loss = F.mse_loss(values / return_scale, returns[mb] / return_scale)
                    mb_valid = aux_valids[mb]
                    if mb_valid.any():
                        aux_pred = self.net.forward_aux(states[mb][mb_valid])
                        aux_loss = F.mse_loss(aux_pred / aux_scale, aux_targets[mb][mb_valid] / aux_scale)
                    else:
                        aux_loss = torch.tensor(0.0, device=self.device)
                    loss = (policy_loss + self.value_coef * value_loss + self.aux_coef * aux_loss
                            - self.entropy_coef * entropy)
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_aux_loss += aux_loss.item()
                    total_entropy += entropy.item()
                    total_clip_frac += clip_frac.item()
                    n_updates += 1

            return (total_policy_loss / n_updates, total_value_loss / n_updates,
                    total_aux_loss / n_updates, total_entropy / n_updates, total_clip_frac / n_updates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000, help="Number of training donnes.")
    parser.add_argument('--batch-episodes', type=int, default=32,
                         help="Number of donnes collected between two PPO updates.")
    parser.add_argument('--epochs', type=int, default=4, help='PPO passes per collected batch.')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--clip-eps', type=float, default=0.2, help='Width of the PPO ratio clip.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--value-coef', type=float, default=0.5, help='Weight of the value loss (MSE).')
    parser.add_argument('--aux-coef', type=float, default=0.5,
                         help="Weight of the auxiliary loss (MSE, normalized) -- only with "
                              "--architecture shared_aux.")
    parser.add_argument('--entropy-coef', type=float, default=0.01, help="Weight of the entropy bonus.")
    parser.add_argument('--entropy-decay', choices=['none', 'invsqrt', 'inv'], default='none',
                         help="Decay of --entropy-coef over episodes (same schedule as "
                              "train.py --entropy-decay): 'none' (constant), 'invsqrt' (beta/sqrt(ep)), "
                              "'inv' (beta/ep).")
    parser.add_argument('--eval-every', type=int, default=200, help='Frequency (in episodes) of greedy evaluations.')
    parser.add_argument('--eval-games', type=int, default=100, help='Number of donnes per evaluation.')
    parser.add_argument('--no-counterfactual-baseline', action='store_true',
                         help="Disables the counterfactual (remarques_rl.md point 3): uses the raw "
                              "reward directly as the PPO return.")
    parser.add_argument('--no-critic-baseline', action='store_true',
                         help="Disables subtracting the predicted value from the advantage -- value_net "
                              "is neither called nor trained. Tests whether a per-state critic adds "
                              "anything over simply centering on the batch's mean (cf. "
                              "update_batch). An axis independent of --no-counterfactual-baseline.")
    parser.add_argument('--ablate-points-so-far', action='store_true',
                         help="Ablation study: forces encode_state's `_points_so_far` block to zero "
                              "(actor and critic), to isolate its effect on performance. --load must "
                              "point to an imit.pt generated with the same flag (pretrain.py "
                              "--ablate-points-so-far).")
    parser.add_argument('--architecture', choices=['small', 'big', 'shared', 'shared_aux', 'shared_deep'],
                         default='small',
                         help="'small' = independent CardNet+ValueNet (default). 'big' = independent "
                              "CardNetBig+ValueNet. 'shared' = SharedTrunkActorCritic (step 2 of the "
                              "plan, remarques_rl.md point 6): shared actor/critic trunk "
                              "(167->128->128, actor head 128->64->32). 'shared_aux' = "
                              "SharedTrunkActorCriticAux (rl_experiments_v5): same + auxiliary task "
                              "(number of trumps remaining for the 3 others, cf. --aux-coef). "
                              "'shared_deep' = SharedTrunkActorCriticDeep (2026-07-28 discussion): "
                              "trunk/head rebalancing, a deeper trunk (167->128->128->64, 3 layers) "
                              "and a shorter actor head (64->32, 1 layer) -- covers exactly "
                              "CardNetBig's 4 layers, unlike 'shared' which only reuses 2 of them. "
                              "--ablate-points-so-far not supported with 'shared'/'shared_aux'/'shared_deep'; "
                              "--no-critic-baseline supported with 'shared'/'shared_deep' (not yet "
                              "'shared_aux'). --load then accepts either a native checkpoint of this "
                              "architecture, or a CardNetBig checkpoint (e.g. imit_bignet_ent02.pt, the "
                              "critic/auxiliary parts then start out random).")
    parser.add_argument('--opponent', default='heuristic',
                         help="Opponent(s) at seats 1/3, comma-separated: same semantics as "
                              "train.py --opponent (heuristic and/or weight paths, a pool if several).")
    parser.add_argument('--pfsp', action='store_true', help='Same semantics as train.py --pfsp.')
    parser.add_argument('--pfsp-refresh-every', type=int, default=5000)
    parser.add_argument('--pfsp-temperature', type=float, default=1.0)
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98)
    parser.add_argument('--pfsp-explore-eps', type=float, default=0.0,
                         help='Same semantics as train.py --pfsp-explore-eps (PFSP exploration floor).')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', default=None, help="Path to save the weights at the end of training.")
    parser.add_argument('--load', default=None, help='Path to resume the policy from saved weights.')
    parser.add_argument('--load-value', default=None,
                         help="Path to load value_net separately (pretrain_value.py, or the 'value' "
                              "key of a native PPO checkpoint) -- without which it starts randomly initialized.")
    parser.add_argument('--checkpoint-every', type=int, default=None,
                         help="Saves a checkpoint every N episodes, in addition to --save at the end of training.")
    parser.add_argument('--checkpoint-dir', default=None,
                         help='Directory to save checkpoints in (required with --checkpoint-every).')
    parser.add_argument('--episode-offset', type=int, default=0,
                         help="Offset added to the episode counter (logs, dealer, decay, checkpoint "
                              "names): to resume a previous training run with consistent absolute "
                              "numbering.")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch is required to train a PPOPolicy (see requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    shared_family = ('shared', 'shared_aux', 'shared_deep')
    if args.architecture in shared_family:
        if args.ablate_points_so_far:
            raise SystemExit("--ablate-points-so-far is not supported with "
                              "--architecture shared/shared_aux/shared_deep.")
        if args.no_critic_baseline and args.architecture == 'shared_aux':
            raise SystemExit("--no-critic-baseline is not yet supported with "
                              "--architecture shared_aux (only 'shared'/'shared_deep').")
        # NeuralPolicy (train.py, used by make_opponent_factory to load a
        # frozen pool opponent) calls net(x) -> forward(x) -- SharedTrunkActorCritic
        # and SharedTrunkActorCriticDeep both define forward() as an alias
        # for forward_actor() for this purpose.
        if args.architecture == 'shared_aux':
            net_cls = SharedTrunkActorCriticAux
        elif args.architecture == 'shared_deep':
            net_cls = SharedTrunkActorCriticDeep
        else:
            net_cls = SharedTrunkActorCritic
    else:
        net_cls = CardNetBig if args.architecture == 'big' else CardNet
    opponent_pool = build_opponent_pool(args.opponent, net_cls=net_cls)
    sampler = PFSPSampler(opponent_pool, refresh_every=args.pfsp_refresh_every,
                           temperature=args.pfsp_temperature, ema_beta=args.pfsp_ema_beta,
                           explore_eps=args.pfsp_explore_eps) if args.pfsp else None

    if args.architecture in shared_family:
        if args.architecture == 'shared_aux':
            policy = SharedTrunkAuxPPOPolicy(lr=args.lr, clip_eps=args.clip_eps, value_coef=args.value_coef,
                                              aux_coef=args.aux_coef, entropy_coef=args.entropy_coef,
                                              epochs=args.epochs, minibatch_size=args.minibatch_size)
        else:
            policy = SharedTrunkPPOPolicy(lr=args.lr, clip_eps=args.clip_eps, value_coef=args.value_coef,
                                           entropy_coef=args.entropy_coef, epochs=args.epochs,
                                           minibatch_size=args.minibatch_size,
                                           no_critic_baseline=args.no_critic_baseline, net_cls=net_cls)
    else:
        policy_net_cls = CardNetBig if args.architecture == 'big' else CardNet
        policy = PPOPolicy(lr=args.lr, clip_eps=args.clip_eps, value_coef=args.value_coef,
                            entropy_coef=args.entropy_coef, epochs=args.epochs, minibatch_size=args.minibatch_size,
                            no_critic_baseline=args.no_critic_baseline,
                            ablate_points=args.ablate_points_so_far, policy_net_cls=policy_net_cls)
    if args.load:
        policy.load(args.load)
        print('weights loaded from', args.load)
    if args.load_value:
        if args.architecture in shared_family:
            raise SystemExit("--load-value doesn't make sense with --architecture shared/shared_aux/"
                              "shared_deep (a single network, no separate value_net).")
        policy.load_value(args.load_value)
        print('value_net weights loaded from', args.load_value)

    if args.checkpoint_every and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    window = []
    adv_window = []
    last_stats = None
    start = time.perf_counter()
    for ep in range(1, args.episodes + 1):
        global_ep = ep + args.episode_offset
        policy.entropy_coef = entropy_beta_for_episode(args.entropy_coef, global_ep, args.entropy_decay)
        dealer = global_ep % 4
        if sampler is not None:
            opponent_name, opponent_factory = sampler.choose(global_ep)
        else:
            opponent_name, opponent_factory = (
                opponent_pool[0] if len(opponent_pool) == 1 else random.choice(opponent_pool)
            )
        reward, hands = run_episode(policy, dealer=dealer, opponent_factory=opponent_factory)
        if sampler is not None:
            sampler.record_outcome(opponent_name, reward)
        if args.no_counterfactual_baseline:
            training_reward = reward
        else:
            training_reward = reward - counterfactual_reward(hands, dealer=dealer, opponent_factory=opponent_factory)
        policy.end_episode(training_reward)

        window.append(reward)
        adv_window.append(training_reward)
        if len(window) > 200:
            window.pop(0)
            adv_window.pop(0)

        if ep % args.batch_episodes == 0:
            stats = policy.update_batch()
            if stats is not None:
                last_stats = stats

        if ep % args.eval_every == 0:
            avg_train = sum(window) / len(window)
            avg_adv = sum(adv_window) / len(adv_window)
            avg_eval, win_rate = evaluate(policy, args.eval_games, start_dealer=global_ep)
            elapsed = time.perf_counter() - start
            if last_stats is None:
                stats_str = ""
            elif len(last_stats) == 5:
                # SharedTrunkAuxPPOPolicy: (policy_loss, value_loss, aux_loss, entropy, clip_frac).
                stats_str = (f"  policy_loss={last_stats[0]:+.4f}  value_loss={last_stats[1]:.4f}  "
                             f"aux_loss={last_stats[2]:.4f}  entropy={last_stats[3]:.3f}  "
                             f"clip_frac={100*last_stats[4]:4.1f}%")
            else:
                stats_str = (f"  policy_loss={last_stats[0]:+.4f}  value_loss={last_stats[1]:.4f}  "
                             f"entropy={last_stats[2]:.3f}  clip_frac={100*last_stats[3]:4.1f}%")
            print(f"episode {global_ep:6d}  train_avg={avg_train:+7.1f}  adv_avg={avg_adv:+7.1f}  "
                  f"eval_avg({args.eval_games})={avg_eval:+7.1f}  win_rate={100*win_rate:5.1f}%  "
                  f"entropy_coef={policy.entropy_coef:.4f}{stats_str}  ({elapsed:.1f}s)")
            if sampler is not None:
                print(f"  picks: {sampler.summary()}")

        if args.checkpoint_every and args.checkpoint_dir and ep % args.checkpoint_every == 0:
            policy.save(os.path.join(args.checkpoint_dir, f'ckpt_ep{global_ep}.pt'))

    # Last partial batch (if args.episodes isn't a multiple of --batch-episodes).
    stats = policy.update_batch()
    if stats is not None:
        last_stats = stats

    if args.save:
        policy.save(args.save)
        print('weights saved to', args.save)

    if sampler is not None:
        print('PFSP picks total:', sampler.summary())


if __name__ == '__main__':
    main()
