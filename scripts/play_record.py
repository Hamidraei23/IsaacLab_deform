"""Record environment 0 of a playback, one CSV row per control step: the action the policy sent and the robot's response.

Used by ``scripts/play.py --record <path.csv>``. The rows are ``teleop_trace.capture_row``'s, the
same columns a ``teleop_ur10e.py`` recording has:

* ``a0``..``a6``: the action handed to ``env.step`` (six twist channels, then the gripper), after
  the runner's clip;
* ``dx``..``drz``: the twist those channels asked for, in metres and radians per step, after the arm
  action's scale (which differs per episode when ``scale_range`` randomises it);
* joint positions, velocities and drive targets, the tool pose in the robot base frame, finger
  half-width, gripper filter and grasp state, each reward term as paid that step, and the
  termination that ended the episode (``ended_by``).

A row is taken at reward time, after the physics of the step and before an ending resets the
environment, so the last row of an episode is the state it ended in. ``step`` counts across
episodes; ``reset_before`` marks each episode's first row. ``<path>.meta.json`` keeps the command
line, the checkpoint and the step length. Works alongside ``--plot`` and the viewer.
"""

import atexit
import sys

FLUSH_EVERY = 50
"""Steps between rewrites of the CSV, so an interrupted run keeps all but the last few."""


def install(path: str, argv: list[str]) -> None:
    """Hook every RSL-RL environment wrapper built from here on to record into ``path``."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from sheet_rl.tasks.manager_based.sheet_twist.teleop_trace import TraceWriter, capture_row

    original_init, original_close = RslRlVecEnvWrapper.__init__, RslRlVecEnvWrapper.close
    command = [sys.argv[0], *argv]  # taken now: the backend swaps sys.argv before the environment is built

    def init(self, env, *args, **kwargs):
        original_init(self, env, *args, **kwargs)
        inner = self.unwrapped
        arm = inner.action_manager.get_term("arm_action")
        terminations, rewards = inner.termination_manager, inner.reward_manager
        trace = TraceWriter(path, {
            "source": "play", "argv": command, "seed": inner.cfg.seed, "num_envs": inner.num_envs,
            "step_dt": inner.step_dt, "action_scale": arm._scale[0].tolist(),
        })
        state = {"step": 0, "episode": 0, "reset_before": True}
        compute = rewards.compute

        def capture(dt):
            reward = compute(dt)
            ended = [name for name in terminations.active_terms if bool(terminations.get_term(name)[0])]
            trace.append(capture_row(
                inner, step=state["step"], episode=state["episode"], action=inner.action_manager.action[0].tolist(),
                twist=arm.processed_actions[0].tolist(), reset_before=state["reset_before"],
                reward=float(reward[0]), ended=ended,
            ))
            state["step"] += 1
            state["reset_before"] = bool(ended)
            if ended:
                trace.flush()
                print(f"[play --record] episode {state['episode']} ended by {ended}; "
                      f"{len(trace.rows)} rows in {trace.path}", flush=True)
                state["episode"] += 1
            elif state["step"] % FLUSH_EVERY == 0:
                trace.flush()
            return reward

        rewards.compute = capture
        self._play_trace = trace
        atexit.register(trace.flush)
        print(f"[play --record] recording environment 0 to {path}", flush=True)

    def close(self):
        trace = getattr(self, "_play_trace", None)
        if trace is not None:
            trace.flush()
        return original_close(self)

    RslRlVecEnvWrapper.__init__, RslRlVecEnvWrapper.close = init, close
