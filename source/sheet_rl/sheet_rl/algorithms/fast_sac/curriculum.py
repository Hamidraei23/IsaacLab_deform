"""Success-gated curricula on environment parameters, stepped between logging windows.

``ParamCurriculum`` raises one parameter of one event term (e.g. the sheet's yaw jitter) by a fixed
step whenever the logged success has held above a threshold for several consecutive windows. The
event reads its ``params`` at call time, so the new value takes effect from the next resets.
"""

from __future__ import annotations

from collections import deque


class ParamCurriculum:
    """Widen ``event_term.params[param]`` by ``step`` once success stays >= ``threshold``.

    Args:
        term_name: the event term (e.g. ``"reset_layout"``).
        param: the parameter in its ``params`` dict (e.g. ``"sheet_yaw_jitter"``).
        start: the value to begin at (written at ``attach``).
        step: the increment per promotion.
        maximum: never exceed this.
        threshold: success (``Episode_Termination/placed`` of a window) required.
        windows: consecutive logging windows that must all be >= threshold (the window right after
            a probe or reset is inflated by synchronised episode starts, so one window is not enough).
        min_iterations_between: cool-down after a promotion, in iterations.
        metric: the logged scalar to gate on.
        unit: multiply by this for logging (e.g. 180/pi to log degrees).
    """

    def __init__(self, term_name: str, param: str, start: float, step: float, maximum: float,
                 threshold: float = 0.75, windows: int = 3, min_iterations_between: int = 300,
                 metric: str = "Episode_Termination/placed", unit: float = 1.0):
        self.term_name = term_name
        self.param = param
        self.value = float(start)
        self.step = float(step)
        self.maximum = float(maximum)
        self.threshold = threshold
        self.windows = windows
        self.min_iterations_between = min_iterations_between
        self.metric = metric
        self.unit = unit
        self._recent: deque[float] = deque(maxlen=windows)
        self._last_promotion = None
        self.promotions: list[tuple[int, float]] = []
        self._cfg = None

    def attach(self, env) -> None:
        """Bind to the environment's event term and write the starting value."""
        self._cfg = env.unwrapped.event_manager.get_term_cfg(self.term_name)
        self._cfg.params[self.param] = self.value

    def update(self, scalars: dict[str, float], iteration: int) -> bool:
        """Call once per logging window; returns True when the parameter was just raised."""
        if self.metric not in scalars:
            return False
        self._recent.append(scalars[self.metric])
        if self._last_promotion is None:
            self._last_promotion = iteration  # cool-down also applies to the first step
        ready = (
            len(self._recent) == self.windows
            and all(v >= self.threshold for v in self._recent)
            and iteration - self._last_promotion >= self.min_iterations_between
            and self.value < self.maximum - 1e-9
        )
        if not ready:
            return False
        self.value = min(self.value + self.step, self.maximum)
        self._cfg.params[self.param] = self.value
        self._recent.clear()
        self._last_promotion = iteration
        self.promotions.append((iteration, self.value))
        return True

    def log_name(self) -> str:
        return f"Curriculum/{self.param}" + ("_deg" if abs(self.unit - 57.29577951308232) < 1e-6 else "")

    def logged_value(self) -> float:
        return self.value * self.unit
