"""Entraine par PPO (Schulman et al. 2017) une policy actor-critic de jeu de
la carte, alternative a REINFORCE (train.py) sur exactement la meme tache :
meme etat/action (`coinche.rl_agent.encode_state`, 32 slots canoniques),
meme equipe RL aux sieges 0/2 partageant les poids, meme adversaire(s) aux
sieges 1/3 (--opponent, pool uniforme ou PFSP), meme contre-factuel du point
3 (`train.counterfactual_reward`). Fichier volontairement separe de train.py
pour pouvoir developper/tester PPO sans risquer d'affecter une experience
REINFORCE deja lancee (le code est relu au demarrage de chaque process, pas
"a chaud" -- mais autant eviter toute ambiguite sur quel code a produit quel
checkpoint).

Difference cle avec REINFORCE : au lieu d'une seule recompense scalaire par
donne appliquee a la somme des log-probs de la trajectoire avec une baseline
EMA globale (`NeuralPolicy.update`), PPO :
  1) accumule un batch de plusieurs donnes (`--batch-episodes`) avant de
     mettre a jour ;
  2) utilise un critique (`ValueNet`) qui apprend a predire le retour attendu
     DEPUIS CHAQUE ETAT de la trajectoire -- baseline plus fine qu'une moyenne
     mobile globale, avantage = retour de la donne moins cette valeur predite ;
  3) fait plusieurs passes (`--epochs`) de descente en minibatches sur ce
     batch, avec un ratio de probabilite clippe (`--clip-eps`) entre la
     policy courante et celle qui a collecte la trajectoire, ce qui autorise
     plusieurs mises a jour par episode collecte sans diverger (contrairement
     a REINFORCE, une mise a jour "on-policy" stricte par episode).
Le retour utilise pour chaque timestep d'une donne est le meme scalaire pour
toute la donne (recompense terminale, cf. contre-factuel du point 3) : il n'y
a pas de recompense intermediaire par pli definie dans ce jeu.

`ValueNet` a son propre tronc, INDEPENDANT de celui de la policy (`CardNet`,
partage avec REINFORCE/imit.pt) : un tronc partage herite de imit.pt (jamais
entraine pour la valeur) laissait le critic quasi incapable de separer
attaque/defense meme apres 50k episodes (cf. remarques_rl.md) -- pre-entrainer
`ValueNet` par regression supervisee du retour final (`pretrain_value.py`,
analogue value de `pretrain.py`) avant le fine-tuning PPO est donc recommande.

`ValueNet` est aussi un critic CENTRALISE (`encode_full_state`, cf.
rl_agent.py) : contrairement a la policy, qui ne voit que ce qu'un joueur
reel verrait, le critic recoit en plus les mains actuelles des 3 autres
sieges -- info privilegiee, connue seulement parce qu'on simule la donne en
entier a l'entrainement, jamais disponible a l'acteur en jeu reel. Motive
par le plafond bas mesure pour un critic a info partielle (R2~0.04-0.06
meme bien entraine, cf. remarques_rl.md) : la variance de ce jeu est
dominee par de l'information cachee qu'aucun critic conditionne sur un etat
partiel ne peut deviner, mais rien n'empeche de lever ce plafond pour le
critic puisqu'il est deja un reseau independant de la policy.

Usage:
    python train_ppo.py --episodes 5000 --eval-every 200
    python train_ppo.py --episodes 2000 --load poids.pt --save poids.pt
    python train_ppo.py --load imit.pt --load-value value_imit.pt --episodes 50000
    python train_ppo.py --load poids.pt --opponent heuristic,100k.pt,250k.pt --pfsp --episodes 100000
"""
import argparse
import os
import random
import time

import torch.nn as nn
import torch.nn.functional as F

from coinche.rl_agent import (
    STATE_DIM, FULL_STATE_DIM, CardNet, encode_state, encode_full_state,
    _canonical_slots, _slot_index, torch,
)
from train import (
    build_opponent_pool, PFSPSampler, run_episode, counterfactual_reward,
    evaluate, entropy_beta_for_episode,
)


if torch is not None:
    class ValueNet(nn.Module):
        """Tronc INDEPENDANT de la policy (pas de partage avec CardNet), 1 couche
        cachee -> 1 scalaire. Avant separation, value_head partageait le tronc
        `fc1` de la policy (herite de imit.pt, jamais entraine pour la valeur) :
        diagnostic (remarques_rl.md) montrant qu'apres 50k episodes PPO le
        critic ne separait quasiment pas attaque/defense (is_attacker, pourtant
        une feature d'entree directe) alors que l'ecart reel de retour entre
        les deux est enorme -- le gradient de value_loss sur ce tronc partage
        etait ecrase par celui de policy_loss. Un tronc dedie, pre-entrainable
        independamment (pretrain_value.py), evite cette concurrence.

        Prend en entree FULL_STATE_DIM (encode_full_state), pas STATE_DIM :
        critic CENTRALISE, qui voit aussi les mains des 3 autres sieges (info
        privilegiee, connue seulement en simulation d'entrainement, jamais par
        l'acteur en jeu reel -- cf. remarques_rl.md, motive par le plafond bas
        du critic a info partielle : R2~0.04-0.06 meme bien entraine, faute de
        pouvoir deviner les mains adverses). Rien n'empeche cette asymetrie
        info-partielle-pour-l'acteur / info-complete-pour-le-critic puisque
        les deux sont des reseaux independants."""
        def __init__(self, in_dim=FULL_STATE_DIM, hidden=128):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = F.relu(self.fc1(x))
            return self.value_head(h).squeeze(-1)


    class PPOPolicy:
        """Interface compatible RLPlayer/run_episode/evaluate (choose_card,
        record, save, load) pour reutiliser telles quelles les fonctions de
        train.py. `record=True` (par defaut) echantillonne et memorise
        (etat visible par l'acteur, etat centralise pour le critic, log-prob,
        valeur, masque des coups legaux) dans `self._traj` ; `record=False`
        (evaluation) joue en glouton (argmax) sans rien memoriser."""
        def __init__(self, device='cpu', lr=1e-3, clip_eps=0.2, value_coef=0.5,
                     entropy_coef=0.01, epochs=4, minibatch_size=64, no_critic_baseline=False,
                     ablate_points=False):
            self.device = device
            self.policy_net = CardNet().to(device)
            self.value_net = ValueNet().to(device)
            self.optimizer = torch.optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
            self.clip_eps = clip_eps
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.epochs = epochs
            self.minibatch_size = minibatch_size
            # Si True : avantage = retour (centre/normalise sur la moyenne du BATCH courant,
            # cf. update_batch), value_net ni utilisee ni entrainee -- teste si un critic par
            # etat apporte quoi que ce soit par rapport a un simple scalaire type REINFORCE.
            self.no_critic_baseline = no_critic_baseline
            # Etude d'ablation : force a zero le bloc `_points_so_far` de encode_state (des
            # deux cotes, acteur et critic) -- cf. rl_experiments_v3/README.md.
            self.ablate_points = ablate_points
            self.record = True
            self._traj = []       # (etat, etat_complet, action_idx, log_prob, valeur, masque) de la donne en cours
            self._episodes = []   # [(traj, retour)] accumules depuis la derniere update_batch()

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
                # Etat centralise (mains adverses incluses) uniquement pour le critic,
                # jamais pour la policy ci-dessus -- cf. docstring de ValueNet.
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
            """A appeler une fois la donne terminee (au lieu de `.update()`
            pour REINFORCE) : archive la trajectoire avec son retour (le
            reward contre-factuel, comme REINFORCE) pour la prochaine
            `update_batch()`. Ne declenche aucune mise a jour immediate."""
            if not self._traj:
                return
            self._episodes.append((self._traj, reward))
            self._traj = []

        def update_batch(self):
            """Mise a jour PPO sur toutes les donnes accumulees depuis le
            dernier appel (cf. `end_episode`) : avantage = retour de la donne
            moins la valeur predite a cet etat (meme retour pour tous les
            timesteps d'une donne, recompense terminale), normalise, puis
            `self.epochs` passes en minibatches sur l'objectif clippe + perte
            de valeur (MSE) + bonus d'entropie. Vide le batch apres coup.
            Si `self.no_critic_baseline` : pas de soustraction de valeur
            predite -- l'avantage normalise ci-dessous devient alors un
            centrage sur la moyenne du BATCH courant (type REINFORCE, mais
            scalaire par batch plutot que EMA globale), `value_net` n'est ni
            appelee ni entrainee (`value_loss` renvoyee a 0).
            Retourne (policy_loss, value_loss, entropy, clip_frac) moyens
            (clip_frac : fraction des echantillons ou |ratio-1| > clip_eps,
            diagnostic standard PPO -- mesure seulement, ne change rien au
            comportement), ou None si rien n'etait accumule."""
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
            # Echelle des retours bruts (~centaines de points, cf. rl_experiments_v2/README.md) :
            # sans ca, value_loss (MSE sur ces retours) domine policy_loss (sur avantage normalise,
            # O(1)) de plusieurs ordres de grandeur -- policy_net et value_net sont desormais des
            # troncs independants (plus de risque de "polluer" les features de la policy), mais
            # `--value-coef` resterait sans effet interpretable sans cette mise a l'echelle commune.
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
                    # Reappliquer le masque des coups legaux (choose_card) : sans lui, new_log_probs
                    # est calcule sous une distribution sur les 32 slots (dont des coups illegaux),
                    # differente de celle sous laquelle old_log_probs a ete echantillonne -- le ratio
                    # PPO ci-dessous n'aurait alors plus le sens "combien la policy a-t-elle bouge".
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
                    # Mesure seule (n'affecte pas loss/gradient) : a quelle frequence le clip
                    # est-il reellement en jeu (|ratio-1| > clip_eps), convention standard PPO.
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
            """Checkpoint natif PPO : les deux reseaux independants dans un seul
            fichier (dict a 2 cles). Pour un resume complet, repasser ce meme
            chemin a la fois a --load et --load-value (chacun y prend sa part).
            Ne sauvegarde PAS l'etat de self.optimizer (moments Adam m/v) : une
            reprise (--load/--load-value puis poursuite de l'entrainement dans
            un nouveau process) repart avec un Adam "a froid" sur des poids
            deja entraines, pas un vrai historique continu -- meme limite deja
            presente pour NeuralPolicy/REINFORCE (train.py) entre les segments
            de rl_experiments_v2/exp_growing_pool. Impact estime faible (les
            moments se re-stabilisent en quelques centaines de steps, une
            fraction negligeable d'un segment de dizaines de milliers
            d'episodes) mais reel a chaque frontiere --episode-offset."""
            torch.save({'policy': self.policy_net.state_dict(),
                        'value': self.value_net.state_dict()}, path)

        def load(self, path):
            """Charge la policy depuis : un checkpoint PPO natif (dict, cle
            'policy'), un checkpoint REINFORCE (CardNet : fc1/fc2, cf.
            rl_agent.py -- imit.pt ou segments de rl_experiments(_v2)/, meme
            forme que policy_net donc aucun remappage requis), ou un ancien
            checkpoint PPO a tronc partage (fc1/policy_head/value_head, avant
            la separation des reseaux -- value_head est alors ignoree)."""
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
                raise ValueError(f"Format de checkpoint non reconnu pour la policy : {path}")

        def load_value(self, path):
            """Charge value_net depuis : un checkpoint PPO natif (dict, cle
            'value'), ou un checkpoint value pre-entraine par pretrain_value.py
            (meme forme que ValueNet, aucun remappage requis)."""
            obj = torch.load(path, map_location=self.device)
            self.value_net.load_state_dict(obj['value'] if 'value' in obj else obj)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=2000, help="Nombre de donnes d'entrainement.")
    parser.add_argument('--batch-episodes', type=int, default=32,
                         help="Nombre de donnes collectees entre deux mises a jour PPO.")
    parser.add_argument('--epochs', type=int, default=4, help='Passes PPO par batch collecte.')
    parser.add_argument('--minibatch-size', type=int, default=64)
    parser.add_argument('--clip-eps', type=float, default=0.2, help='Largeur du clip du ratio PPO.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--value-coef', type=float, default=0.5, help='Poids de la perte de valeur (MSE).')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help="Poids du bonus d'entropie.")
    parser.add_argument('--entropy-decay', choices=['none', 'invsqrt', 'inv'], default='none',
                         help="Decroissance de --entropy-coef au fil des episodes (meme schema que "
                              "train.py --entropy-decay) : 'none' (constant), 'invsqrt' (beta/sqrt(ep)), "
                              "'inv' (beta/ep).")
    parser.add_argument('--eval-every', type=int, default=200, help='Frequence (en episodes) des evaluations gloutonnes.')
    parser.add_argument('--eval-games', type=int, default=100, help='Nombre de donnes par evaluation.')
    parser.add_argument('--no-counterfactual-baseline', action='store_true',
                         help="Desactive le contre-factuel (remarques_rl.md point 3) : utilise le reward "
                              "brut directement comme retour PPO.")
    parser.add_argument('--no-critic-baseline', action='store_true',
                         help="Desactive la soustraction de la valeur predite dans l'avantage -- value_net "
                              "n'est ni appelee ni entrainee. Teste si un critic par etat apporte quoi que "
                              "ce soit par rapport a un simple centrage sur la moyenne du batch (cf. "
                              "update_batch). Axe independant de --no-counterfactual-baseline.")
    parser.add_argument('--ablate-points-so-far', action='store_true',
                         help="Etude d'ablation : force a zero le bloc `_points_so_far` de encode_state "
                              "(acteur et critic), pour isoler son effet sur la performance. --load doit "
                              "pointer vers un imit.pt genere avec le meme flag (pretrain.py "
                              "--ablate-points-so-far).")
    parser.add_argument('--opponent', default='heuristic',
                         help="Adversaire(s) aux sieges 1/3, separes par des virgules : meme semantique que "
                              "train.py --opponent (heuristic et/ou chemins de poids, pool si plusieurs).")
    parser.add_argument('--pfsp', action='store_true', help='Meme semantique que train.py --pfsp.')
    parser.add_argument('--pfsp-refresh-every', type=int, default=5000)
    parser.add_argument('--pfsp-temperature', type=float, default=1.0)
    parser.add_argument('--pfsp-ema-beta', type=float, default=0.98)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save', default=None, help="Chemin pour sauvegarder les poids en fin d'entrainement.")
    parser.add_argument('--load', default=None, help='Chemin pour reprendre la policy depuis des poids sauvegardes.')
    parser.add_argument('--load-value', default=None,
                         help="Chemin pour charger value_net separement (pretrain_value.py, ou cle 'value' "
                              "d'un checkpoint PPO natif) -- sans quoi elle demarre initialisee aleatoirement.")
    parser.add_argument('--checkpoint-every', type=int, default=None,
                         help="Sauvegarde un checkpoint tous les N episodes, en plus de --save en fin d'entrainement.")
    parser.add_argument('--checkpoint-dir', default=None,
                         help='Dossier de sauvegarde des checkpoints (requis avec --checkpoint-every).')
    parser.add_argument('--episode-offset', type=int, default=0,
                         help="Decalage ajoute au compteur d'episode (logs, dealer, decroissance, nom des "
                              "checkpoints) : pour reprendre un entrainement precedent avec une numerotation "
                              "absolue coherente.")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("torch est requis pour entrainer une PPOPolicy (voir requirements.txt).")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    opponent_pool = build_opponent_pool(args.opponent)
    sampler = PFSPSampler(opponent_pool, refresh_every=args.pfsp_refresh_every,
                           temperature=args.pfsp_temperature, ema_beta=args.pfsp_ema_beta) if args.pfsp else None

    policy = PPOPolicy(lr=args.lr, clip_eps=args.clip_eps, value_coef=args.value_coef,
                        entropy_coef=args.entropy_coef, epochs=args.epochs, minibatch_size=args.minibatch_size,
                        no_critic_baseline=args.no_critic_baseline,
                        ablate_points=args.ablate_points_so_far)
    if args.load:
        policy.load(args.load)
        print('poids charges depuis', args.load)
    if args.load_value:
        policy.load_value(args.load_value)
        print('poids de value_net charges depuis', args.load_value)

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
            stats_str = (f"  policy_loss={last_stats[0]:+.4f}  value_loss={last_stats[1]:.4f}  "
                         f"entropy={last_stats[2]:.3f}  clip_frac={100*last_stats[3]:4.1f}%"
                         ) if last_stats is not None else ""
            print(f"episode {global_ep:6d}  train_avg={avg_train:+7.1f}  adv_avg={avg_adv:+7.1f}  "
                  f"eval_avg({args.eval_games})={avg_eval:+7.1f}  win_rate={100*win_rate:5.1f}%  "
                  f"entropy_coef={policy.entropy_coef:.4f}{stats_str}  ({elapsed:.1f}s)")
            if sampler is not None:
                print(f"  picks: {sampler.summary()}")

        if args.checkpoint_every and args.checkpoint_dir and ep % args.checkpoint_every == 0:
            policy.save(os.path.join(args.checkpoint_dir, f'ckpt_ep{global_ep}.pt'))

    # Dernier batch partiel (si args.episodes n'est pas multiple de --batch-episodes).
    stats = policy.update_batch()
    if stats is not None:
        last_stats = stats

    if args.save:
        policy.save(args.save)
        print('poids sauvegardes dans', args.save)

    if sampler is not None:
        print('PFSP picks total:', sampler.summary())


if __name__ == '__main__':
    main()
