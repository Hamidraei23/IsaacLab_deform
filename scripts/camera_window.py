"""Windows that show image streams while a simulation plays (``play_actor.py --camera``, ``--depth``).

Tk with Pillow's ``ImageTk``, like the repo's other GUIs: the venv's OpenCV is a headless build with no
``imshow``. All windows share one hidden Tk root (a second ``tk.Tk()`` would be a second interpreter, and an image
made in one cannot be drawn in the other: ``image "pyimage3" does not exist``); each window is a ``Toplevel``.
``show`` is called once per control step from the play loop and returns False once the window has been closed (or q
pressed in it), so the loop can stop.
"""

import tkinter as tk

from PIL import Image, ImageTk

_root = None


def _tk_root() -> tk.Tk:
    global _root
    if _root is None:
        _root = tk.Tk()
        _root.withdraw()
    return _root


class CameraWindow:
    def __init__(self, title: str, width: int, height: int):
        self.root = _tk_root()
        self.top = tk.Toplevel(self.root)
        self.top.title(title)
        self.top.geometry(f"{width}x{height}")
        self.label = tk.Label(self.top, bg="black")
        self.label.pack(fill="both", expand=True)
        self.open = True
        self.photo = None
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        self.top.bind("<q>", lambda _e: self.close())

    def show(self, frame) -> bool:
        """Display an (H, W, 3) or (H, W, 4) uint8 RGB(A) array. Returns False once the window is closed."""
        if not self.open:
            return False
        image = Image.fromarray(frame[..., :3])
        w, h = self.label.winfo_width(), self.label.winfo_height()
        if w > 1 and h > 1 and (w, h) != image.size:  # the user resized the window: fit, keeping the aspect
            scale = min(w / image.width, h / image.height)
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(image, master=self.top)  # kept on self: Tk only holds a weak reference
        self.label.configure(image=self.photo)
        try:
            self.root.update()
        except tk.TclError:  # destroyed from the window manager mid-update
            self.open = False
        return self.open

    def close(self) -> None:
        if self.open:
            self.open = False
            self.top.destroy()
