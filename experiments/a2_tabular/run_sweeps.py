"""A2 item 3: sweep the discount factor and the learning rate.

The two parameters are studied separately because they do different jobs:

* gamma changes *what the optimal policy is*, so it is swept with DP on a
  corridor MDP where the switching point can be derived by hand.
* alpha changes *how the estimate is tracked*, so it is swept with TD control on
  cliff walking, deterministic and slippery, where its stability role shows up.
"""

from __future__ import annotations

import argparse

import numpy as np
from gridworld import CLIFF, GridWorld
from mdp import TabularMDP, value_iteration
from td import QLearning, Sarsa, train

from rl_ladder import Run, seed_everything

GAMMAS = [0.1, 0.3, 0.5, 0.55, 0.56, 0.57, 0.6, 0.8, 0.95, 0.99]
ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]

NEAR_REWARD, FAR_REWARD, CORRIDOR_LEN = 1.0, 10.0, 5


def corridor_mdp() -> TabularMDP:
    """Near +1 in one step, or far +10 after `CORRIDOR_LEN` steps.

    States `0..CORRIDOR_LEN-1` are the corridor, then two terminals. Action 0 is
    'take the near reward', action 1 is 'walk on'; inside the corridor both
    actions just advance, so the only real decision is at state 0.
    """
    near_terminal, far_terminal = CORRIDOR_LEN, CORRIDOR_LEN + 1
    n_states = CORRIDOR_LEN + 2
    transitions = np.zeros((n_states, 2, n_states))
    rewards = np.zeros((n_states, 2))

    transitions[0, 0, near_terminal] = 1.0
    rewards[0, 0] = NEAR_REWARD
    transitions[0, 1, 1] = 1.0

    for state in range(1, CORRIDOR_LEN):
        nxt = far_terminal if state == CORRIDOR_LEN - 1 else state + 1
        reward = FAR_REWARD if state == CORRIDOR_LEN - 1 else 0.0
        for action in (0, 1):
            transitions[state, action, nxt] = 1.0
            rewards[state, action] = reward

    for terminal in (near_terminal, far_terminal):
        transitions[terminal, :, terminal] = 1.0

    is_terminal = np.zeros(n_states, dtype=bool)
    is_terminal[[near_terminal, far_terminal]] = True
    return TabularMDP(transitions, rewards, is_terminal)


def gamma_study(run: Run) -> dict:
    """Sweep gamma on the corridor and compare the switch to the analytic threshold."""
    # Q(0, far) = gamma^(CORRIDOR_LEN-1) * FAR, Q(0, near) = NEAR. Equal when:
    threshold = (NEAR_REWARD / FAR_REWARD) ** (1.0 / (CORRIDOR_LEN - 1))
    mdp = corridor_mdp()
    print(f"analytic switch at gamma = (1/10)^(1/4) = {threshold:.4f}")

    choices = {}
    for gamma in GAMMAS:
        v, policy, _ = value_iteration(mdp, gamma)
        q_start = mdp.q_from_v(v, gamma)[0]
        choice = "far" if policy[0] == 1 else "near"
        choices[gamma] = choice
        run.log(step=len(choices), param="gamma", value=gamma, choice=choice,
                q_near=float(q_start[0]), q_far=float(q_start[1]))
        print(f"  gamma={gamma:<5g} Q(near)={q_start[0]:6.3f}  Q(far)={q_start[1]:6.3f}  -> {choice}")

    switched = [g for g in GAMMAS if choices[g] == "far"]
    empirical = min(switched) if switched else None
    correct = all((choices[g] == "far") == (g > threshold) for g in GAMMAS)
    print(f"  first gamma choosing far: {empirical} | matches theory everywhere: {correct}")
    return {"threshold": threshold, "choices": {str(k): v for k, v in choices.items()},
            "first_far_gamma": empirical, "matches_theory": correct}


def alpha_study(run: Run, args) -> dict:
    """Sweep alpha for both TD agents, on deterministic and slippery cliffs."""
    results = {}
    for slip in (0.0, args.slip):
        for cls in (QLearning, Sarsa):
            key = f"{cls.__name__}-slip{slip:g}"
            results[key] = {}
            for alpha in ALPHAS:
                seed_everything(0)
                finals = []
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    env = GridWorld(CLIFF, slip=slip)
                    agent = cls(env.n_states, env.n_actions, alpha=alpha, gamma=1.0, eps=args.eps)
                    returns, _ = train(env, agent, args.episodes, rng)
                    finals.append(returns[-100:].mean())
                mean = float(np.mean(finals))
                results[key][alpha] = mean
                run.log(step=len(results[key]), param="alpha", agent=key, value=alpha, online_return=mean)
            best = max(results[key], key=results[key].get)
            line = "  ".join(f"a={a:g}:{results[key][a]:7.1f}" for a in ALPHAS)
            print(f"{key:20s} {line}   best alpha={best:g}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--slip", type=float, default=0.1)
    args = parser.parse_args()

    run = Run(phase="a2", name="sweeps", config=vars(args))
    print("=== gamma study (DP on the corridor MDP) ===")
    gamma_results = gamma_study(run)
    print("\n=== alpha study (TD control on cliff walking) ===")
    alpha_results = alpha_study(run, args)
    print(f"\nrun -> {run.finish({'gamma': gamma_results, 'alpha': alpha_results})}")


if __name__ == "__main__":
    main()
