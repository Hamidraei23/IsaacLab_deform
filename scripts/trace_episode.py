"""List the episodes in a ``play.py --record`` CSV, or save one of them as its own CSV.

Without ``--episode`` it prints every episode in the recording: its number, length (steps and seconds), how it
ended and its peak tool speed. With ``--episode N`` it writes that episode's rows, every column kept, ``step`` and
``t`` counted from the episode's start, to ``<recording>_ep<N>.csv`` (or ``--out``), with the recording's
``.meta.json`` copied beside it plus the episode number. The last episode is usually cut off by Ctrl-C or the
end of a clip; it is listed as unfinished. ``scripts/trace_episode_gui.py`` does the same in a window, with plots.

Usage (plain Python, no simulator)::

    python scripts/trace_episode.py logs/play/ur16e_velterm45.csv              # list the episodes
    python scripts/trace_episode.py logs/play/ur16e_velterm45.csv --episode 3  # save episode 3
"""

import argparse
import csv
import json
import math
import os


def meta_path(path):
    return os.path.splitext(path)[0] + ".meta.json"


def load_recording(path):
    """(columns, {episode number: rows}, meta) of a recording; rows keep the CSV's text values."""
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    meta = {}
    if os.path.exists(meta_path(path)):
        with open(meta_path(path)) as handle:
            meta = json.load(handle)
    episodes = {}
    for row in rows:
        episodes.setdefault(int(row["episode"]), []).append(row)
    return columns, episodes, meta


def ending(rows):
    return rows[-1]["ended_by"] or "unfinished (recording stopped)"


def tool_speeds(rows, dt):
    """Tool-point speed [m/s] between consecutive steps."""
    tool = [tuple(float(r[k]) for k in ("tool_x", "tool_y", "tool_z")) for r in rows]
    return [math.dist(a, b) / dt for a, b in zip(tool, tool[1:])]


def save_episode(path, columns, episodes, meta, number, out=None):
    """Write episode ``number`` (``step`` and ``t`` from its start) and its meta; returns the CSV path."""
    rows = episodes[number]
    first_step, first_t = int(rows[0]["step"]), float(rows[0]["t"])
    out = out or f"{os.path.splitext(path)[0]}_ep{number}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "step": int(row["step"]) - first_step, "t": f"{float(row['t']) - first_t:.4f}"})
    with open(meta_path(out), "w") as handle:
        json.dump({**meta, "source_csv": path, "episode": number}, handle, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="A recording written by play.py --record.")
    parser.add_argument("--episode", type=int, default=None, help="The episode to save (a number from the listing).")
    parser.add_argument("--out", default=None, help="Where to write it (default: <recording>_ep<N>.csv).")
    args = parser.parse_args()
    columns, episodes, meta = load_recording(args.csv)
    dt = float(meta.get("step_dt", 0.02))
    if args.episode is None:
        print(f"{args.csv}: {len(episodes)} episodes, {sum(map(len, episodes.values()))} steps, step {dt * 1000:g} ms")
        print(f"{'episode':>7s} | {'steps':>5s} | {'seconds':>7s} | {'peak tool m/s':>13s} | ended by")
        for number, rows in episodes.items():
            peak = max(tool_speeds(rows, dt), default=0.0)
            print(f"{number:7d} | {len(rows):5d} | {len(rows) * dt:7.2f} | {peak:13.2f} | {ending(rows)}")
        return
    if args.episode not in episodes:
        raise SystemExit(f"no episode {args.episode} in {args.csv}; it has {sorted(episodes)}")
    out = save_episode(args.csv, columns, episodes, meta, args.episode, args.out)
    rows = episodes[args.episode]
    print(f"episode {args.episode}: {len(rows)} steps ({len(rows) * dt:.2f} s), ended by {ending(rows)} -> {out}")


if __name__ == "__main__":
    main()
