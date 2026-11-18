# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-key shortcuts that work in whichever viewer the run happens to have open.

There are two, and they are unrelated to each other in every way except that both are reached by
pressing a key: ``X`` pushes the sheet's lower edge and ``R`` restarts the episode. Neither is part
of the MDP.

**Two viewers, two entirely different keyboard APIs.** ``--viz newton_gl`` renders through Newton's
own OpenGL viewer, which is a pyglet window and takes handlers pushed onto it. ``--viz kit`` renders
through Omniverse Kit, which has no pyglet window at all and instead delivers keystrokes through
carb's input interface, subscribed against the application window. A binder that knows only the
first -- which is what this started as -- goes quiet the moment the viewer is switched, and says so
only after ten seconds of looking for a window that was never going to appear.

Both are attempted, every step, until each has either bound or given up. Binding both when both
exist is deliberate and harmless: keystrokes go to the focused window, so only one of them can fire
for any given press.

**Why the binding is retried rather than done once.** Viewers are built lazily, on the first render,
which is several steps after the environment exists. There is no event to wait for and no cost to
asking again, so :meth:`ViewerKeys.update` asks on every step until it succeeds.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


GIVE_UP_STEPS = 300
"""Steps to keep looking for a viewer before saying the keys are dead [steps].

Ten seconds at the 30 Hz control rate. A headless run, or one on ``--viz rerun``, never finds a
window and should say so once rather than leaving keys that quietly do nothing.
"""


class HeldKeys(Protocol):
    """A consumer of held keys: ``keys`` names them (upper case), the rest are the callbacks."""

    keys: tuple[str, ...]

    def press(self, key: str) -> None: ...

    def release(self, key: str) -> None: ...

    def release_all(self) -> None: ...


class ViewerKeys:
    """Binds ``key -> callback`` on the Newton GL window, the Kit window, or both.

    Args:
        env: The environment, used to find the Newton visualizer's window.
        bindings: What each key should do, keyed by pyglet/carb key name -- ``X``, ``R``, ``F1``.
            Case-insensitive. The callbacks take no arguments and run on the *press*, on the
            viewer's thread, so they should do no more than set a flag or increment a counter.
        held: Optional keys tracked while they are held down (press *and* release), for continuous
            control such as the platform teleop. In Newton GL their presses are consumed, so the
            viewer's own ``WASDQE`` camera flight does not also fire on them.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        bindings: dict[str, Callable[[], None]],
        held: HeldKeys | None = None,
    ) -> None:
        self.env = env
        self.bindings = {key.upper(): callback for key, callback in bindings.items()}
        self.held = held

        self._newton_bound = False
        self._kit_bound = False
        self._attempts = 0
        # Held so the subscription is not collected while the run is alive.
        self._kit_subscription = None

    """
    Operations.
    """

    def update(self) -> None:
        """Bind whatever can be bound now. Call once per step; cheap and idempotent once bound."""
        if self._attempts >= GIVE_UP_STEPS or (self._newton_bound and self._kit_bound):
            return

        self._attempts += 1
        if not self._newton_bound:
            self._newton_bound = self._bind_newton_gl()
        if not self._kit_bound:
            self._kit_bound = self._bind_kit()

        if self._newton_bound or self._kit_bound:
            where = " and ".join(
                name
                for name, bound in (("Newton GL", self._newton_bound), ("Kit", self._kit_bound))
                if bound
            )
            names = sorted(self.bindings) + (sorted(self.held.keys) if self.held is not None else [])
            print(f"[sheet_rl] keys bound in the {where} window: {', '.join(names)}")
            # Stop retrying the other backend: one window is enough, and a run with both would
            # otherwise print again the moment the second one appears.
            self._attempts = GIVE_UP_STEPS
        elif self._attempts == GIVE_UP_STEPS:
            print(
                "[sheet_rl] no viewer window found, so the keyboard shortcuts are inactive."
                " Run with '--viz newton_gl' or '--viz kit' on a machine with a display to use them."
            )

    """
    Internal helpers.
    """

    def _bind_newton_gl(self) -> bool:
        """Push a handler onto Newton's pyglet window. ``False`` while there is no such window."""
        resolved = self._resolve_gl_window()
        if resolved is None:
            return False

        viewer, window = resolved

        import pyglet

        symbols = {}
        for key, callback in self.bindings.items():
            symbol = getattr(pyglet.window.key, key, None)
            if symbol is None:
                print(f"[sheet_rl] '{key}' is not a pyglet key name and is inactive in Newton GL.")
                continue
            symbols[symbol] = callback

        held_symbols = {}
        if self.held is not None:
            for key in self.held.keys:
                symbol = getattr(pyglet.window.key, key, None)
                if symbol is not None:
                    held_symbols[symbol] = key
        held = self.held

        def on_key_press(pressed: int, modifiers: int):
            callback = symbols.get(pressed)
            held_key = held_symbols.get(pressed)
            if callback is None and held_key is None:
                return None
            gui = getattr(viewer, "gui", None)
            # Let ImGui consume text input.
            if gui is not None and gui.is_keyboard_capturing():
                return None
            if held_key is not None:
                # Consumed, so the KeyStateHandler under this never sees it and the camera stays.
                held.press(held_key)
            else:
                callback()
            return pyglet.event.EVENT_HANDLED

        def on_key_release(released: int, modifiers: int):
            held_key = held_symbols.get(released)
            if held_key is not None:
                held.release(held_key)
            # Never consumed: the KeyStateHandler must also see the key go up.
            return None

        def on_deactivate():
            # A key released while the window is unfocused never reports its release.
            if held is not None:
                held.release_all()
            return None

        # Run first and consume the key event.
        window.push_handlers(on_key_press=on_key_press)
        if held_symbols:
            window.push_handlers(on_key_release=on_key_release, on_deactivate=on_deactivate)
        return True

    def _resolve_gl_window(self) -> tuple[object, object] | None:
        """The Newton GL viewer and its pyglet window, or ``None`` while there is no such window.

        Returns ``None`` rather than raising for every reason there might not be one: a headless
        run, a visualizer that is not Newton's, or simply a viewer that has not been built yet --
        it is created lazily on the first render, which is after this is first asked.
        """
        for visualizer in getattr(getattr(self.env, "sim", None), "visualizers", None) or []:
            viewer = getattr(visualizer, "_viewer", None)
            window = getattr(getattr(viewer, "renderer", None), "window", None)
            if window is not None and hasattr(window, "push_handlers"):
                return viewer, window
        return None

    def _bind_kit(self) -> bool:
        """Subscribe to Kit's keyboard through carb. ``False`` when the run has no Kit app window.

        The import itself is the test: ``omni.appwindow`` exists only inside a Kit application, so a
        kitless or headless run fails here rather than on the subscription.
        """
        try:
            import carb
            import omni.appwindow

            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            if keyboard is None:
                return False
            press = carb.input.KeyboardEventType.KEY_PRESS
            release = carb.input.KeyboardEventType.KEY_RELEASE

            def on_event(event, *args, keys=weakref.proxy(self)) -> bool:
                name = event.input.name
                if keys.held is not None and name in keys.held.keys:
                    if event.type == press:
                        keys.held.press(name)
                    elif event.type == release:
                        keys.held.release(name)
                elif event.type == press:
                    callback = keys.bindings.get(name)
                    if callback is not None:
                        callback()
                # Never consume the event: Kit's own camera and UI shortcuts keep working.
                return True

            self._kit_subscription = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
                keyboard, on_event
            )
        except Exception:
            # Any failure here means "no Kit keyboard to bind", which is a normal state -- a Newton
            # GL run, a headless one -- and not something to raise out of a step.
            return False
        return True
