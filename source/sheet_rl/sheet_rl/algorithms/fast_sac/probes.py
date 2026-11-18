"""Probes: short fixed-condition rollouts run between training iterations, for the log only.

A probe takes over the environments, runs its episodes with the current actor, restores what it
changed and returns scalars. Its transitions are not stored; the trainer resets afterwards.
"""

from __future__ import annotations

import torch

from .runner import IsaacLabSacEnv

OUTCOMES = ("placed", "dropped", "held_too_low", "time_out")


class ActionScaleProbe:
    """One (or more) episodes per environment at each fixed IK action scale (user request: every
    few thousand iterations, try 0.05 and 0.1)."""

    def __init__(self, scales: list[float], episodes: int = 1, deterministic: bool = False,
                 action_name: str = "arm_action", max_steps: int | None = 400):
        """``max_steps`` caps each scale's rollout: an episode still running then counts as a failure
        (an unfinished episode is a policy that never released). ``None`` waits for the time-out,
        which one hovering environment can stretch to the full episode limit."""
        self.scales = [float(s) for s in scales]
        self.episodes = episodes
        self.deterministic = deterministic
        self.action_name = action_name
        self.max_steps = max_steps

    @torch.no_grad()
    def run(self, alg, env: IsaacLabSacEnv) -> dict[str, float]:
        from sheet_rl.tasks.manager_based.sheet_twist.sheet_twist_mdp import set_action_scale

        unwrapped = env.unwrapped
        term = unwrapped.action_manager.get_term(self.action_name)
        saved_scale = term._scale.clone()
        # the reset event would redraw the scale on every reset; hold it off while probing
        event_cfg = None
        try:
            event_cfg = unwrapped.event_manager.get_term_cfg("randomize_arm_scale")
            saved_range = event_cfg.params.get("scale_range")
            event_cfg.params["scale_range"] = None
        except (ValueError, KeyError):
            saved_range = None
        out: dict[str, float] = {}
        max_steps = self.max_steps if self.max_steps else self.episodes * env.max_episode_length + 10
        try:
            for scale in self.scales:
                env.reset()
                set_action_scale(unwrapped, scale, self.action_name)  # after the reset's own event
                obs, _ = env._split(unwrapped.obs_buf)
                n = env.num_envs
                done_count = torch.zeros(n, dtype=torch.long, device=env.device)
                ep_return = torch.zeros(n, device=env.device)
                ep_length = torch.zeros(n, device=env.device)
                returns, lengths, outcomes = [], [], {name: 0 for name in (*OUTCOMES, "other")}
                for _ in range(max_steps):
                    if not (done_count < self.episodes).any():
                        break
                    step = env.step(alg.act(obs, deterministic=self.deterministic))
                    set_action_scale(unwrapped, scale, self.action_name)  # resets inside the step restore the cfg scale
                    ep_return += step["rewards"]
                    ep_length += 1
                    done = step["dones"]
                    if done.any():
                        flags = env.termination_flags(OUTCOMES)
                        for i in done.nonzero(as_tuple=False).flatten().tolist():
                            if done_count[i] < self.episodes:
                                outcomes[next((k for k in OUTCOMES if k in flags and bool(flags[k][i])), "other")] += 1
                                returns.append(ep_return[i].item())
                                lengths.append(ep_length[i].item())
                                done_count[i] += 1
                        ep_return[done] = 0.0
                        ep_length[done] = 0.0
                    obs = step["obs"]
                expected = n * self.episodes  # every environment owes this many episodes
                unfinished = expected - len(returns)
                m = max(len(returns), 1)  # means over the episodes that did end
                tag = f"scale_{scale:g}"
                out[f"Probe/{tag}/episodes"] = float(len(returns))
                out[f"Probe/{tag}/unfinished"] = float(unfinished)
                # rates over the episodes owed: an episode that did not end within max_steps is a failure
                out[f"Probe/{tag}/success"] = outcomes["placed"] / expected
                out[f"Probe/{tag}/dropped"] = outcomes["dropped"] / expected
                out[f"Probe/{tag}/return"] = sum(returns) / m if returns else float("nan")
                out[f"Probe/{tag}/length"] = sum(lengths) / m if lengths else float("nan")
        finally:
            term._scale[:] = saved_scale
            if event_cfg is not None:
                event_cfg.params["scale_range"] = saved_range
        return out
