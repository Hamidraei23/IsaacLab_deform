"""Draw the nine sheet landmarks the policy observes (observations.sheet_key_points) and their distances.

Reads sheet_points_at_reset.json (written from the UR16e task right after a reset: the 81 mesh nodes and the nine
landmarks in the env and robot frames) and writes sheet_landmarks.png and sheet_landmarks.md next to it.

    python evaluations/sheet_landmarks/plot_sheet_landmarks.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "sheet_points_at_reset.json")))
nodes = np.array(data["all_nodes_env"])
land = np.array(data["env_frame"])
robot = np.array(data["robot_frame"])
picks, ids = data["picks"], data["node_ids"]
cols = data["resolution"][0] + 1

# the sheet's own 2-D frame: u along the bottom edge (node 0 -> node 8), v up the sheet (node 0 -> node 72)
origin = nodes[0]
u_axis = nodes[cols - 1] - origin
u_axis /= np.linalg.norm(u_axis)
v_axis = nodes[-cols] - origin
v_axis -= (v_axis @ u_axis) * u_axis
v_axis /= np.linalg.norm(v_axis)
to_uv = lambda p: np.stack([(p - origin) @ u_axis, (p - origin) @ v_axis], axis=-1)  # noqa: E731
nodes_uv, land_uv = to_uv(nodes) * 100, to_uv(land) * 100  # cm
dist = np.linalg.norm(land[:, None] - land[None], axis=-1) * 100  # cm

names = ["bottom-left corner", "bottom-edge middle", "bottom-right corner", "left-edge middle", "centre",
         "right-edge middle", "top-left corner", "top-edge middle", "top-right corner"]

fig = plt.figure(figsize=(14, 7.2))
ax = fig.add_axes([0.04, 0.08, 0.46, 0.84])
ax.scatter(nodes_uv[:, 0], nodes_uv[:, 1], s=6, color="#b8b8b8", zorder=1, label="mesh node (81, every 2.5 cm)")
ax.add_patch(plt.Rectangle((0, 0), 20, 20, fill=False, edgecolor="#888888", lw=1, zorder=0))
pairs = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)]
for a, b in pairs:
    pa, pb = land_uv[a], land_uv[b]
    ax.plot(*zip(pa, pb), color="#3b6fb6", lw=1.6, zorder=2)
    mid = (pa + pb) / 2
    horizontal = abs(pa[1] - pb[1]) < 1e-6
    ax.text(mid[0] + (0 if horizontal else 0.5), mid[1] + (0.6 if horizontal else 0), f"{dist[a, b]:.1f} cm",
            ha="center" if horizontal else "left", va="bottom" if horizontal else "center", fontsize=9, color="#3b6fb6",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=5)
for a, b in ((0, 4), (4, 8)):
    pa, pb = land_uv[a], land_uv[b]
    ax.plot(*zip(pa, pb), color="#c0782a", lw=1.1, ls="--", zorder=2)
    mid = (pa + pb) / 2
    ax.text(mid[0] + 0.6, mid[1] - 0.6, f"{dist[a, b]:.2f} cm", fontsize=9, color="#c0782a", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=5)
ax.plot(*zip(land_uv[0], land_uv[8]), color="#c0782a", lw=0.8, ls=":", zorder=1)
ax.text(13.2, 12.0, f"0 to 8: {dist[0, 8]:.2f} cm", fontsize=9, color="#c0782a", rotation=45, ha="center")
ax.scatter(land_uv[:, 0], land_uv[:, 1], s=140, color="#d6453d", zorder=3, edgecolor="white", lw=1.5)
for i, (u, v) in enumerate(land_uv):
    ax.text(u - 0.9, v + 0.9, str(i), fontsize=13, fontweight="bold", color="#d6453d", ha="right", va="bottom", zorder=4)
    ax.text(u + 0.9, v - 0.9, f"node {ids[i]}", fontsize=7.5, color="#666666", ha="left", va="top", zorder=4)
ax.axhspan(-3, 0.4, color="#e9e2d0", zorder=0)
ax.text(10, -2.1, "bottom edge: inside the slot at reset (z = 0.002 m)", ha="center", fontsize=8.5, color="#7a6a40")
ax.annotate("", xy=(23.5, 0), xytext=(23.5, 20), arrowprops=dict(arrowstyle="<->", color="#888"))
ax.text(24.2, 10, "20 cm", rotation=90, va="center", color="#666")
ax.annotate("", xy=(0, 23.2), xytext=(20, 23.2), arrowprops=dict(arrowstyle="<->", color="#888"))
ax.text(10, 23.8, "20 cm", ha="center", color="#666")
ax.set_xlim(-3, 27)
ax.set_ylim(-3.5, 26)
ax.set_aspect("equal")
ax.set_xlabel("along the sheet, u [cm]  (bottom edge, node 0 to node 8)")
ax.set_ylabel("up the sheet, v [cm]")
ax.set_title("The 9 observed sheet points, sheet seen face-on at reset", fontsize=12)
ax.legend(loc="upper right", frameon=False, fontsize=8)

# the distance table
tx = fig.add_axes([0.53, 0.08, 0.45, 0.84])
tx.axis("off")
cell = [[("" if i == j else f"{dist[i, j]:.1f}") for j in range(9)] for i in range(9)]
table = tx.table(cellText=cell, rowLabels=[f"{i}" for i in range(9)], colLabels=[f"{j}" for j in range(9)],
                 loc="upper center", cellLoc="center", bbox=[0.04, 0.52, 0.94, 0.42])
table.auto_set_font_size(False)
table.set_fontsize(9)
for (r, c), t in table.get_celld().items():
    t.set_edgecolor("#cccccc")
    if r == 0 or c == -1:
        t.set_text_props(fontweight="bold", color="#d6453d")
        t.set_facecolor("#f6f6f6")
tx.text(0.51, 0.97, "Distance between every pair of observed points [cm], flat sheet at reset", ha="center",
        fontsize=11, transform=tx.transAxes)
rows = ["index  where                 node   robot frame x, y, z [m]"]
for i in range(9):
    rows.append(f"{i:>3}    {names[i]:<21} {ids[i]:>4}   {robot[i, 0]:+.3f}, {robot[i, 1]:+.3f}, {robot[i, 2]:+.3f}")
tx.text(0.04, 0.47, "\n".join(rows), family="monospace", fontsize=8.6, va="top", transform=tx.transAxes)
tx.text(0.04, 0.06,
        "Grid: 8 x 8 elements, 9 x 9 nodes, 2.5 cm apart. The observed points are nodes (row, col) with rows and\n"
        "cols in {0, 4, 8}: neighbours 10 cm apart, diagonal neighbours 14.1 cm, corner to corner 20 / 28.3 cm.\n"
        "The policy reads each as an offset from the tool point (robot-frame axes), 27 values; the robot-frame\n"
        "positions above are the UR16e task's fixed sheet pose (yaw -0.54 rad) right after a reset.",
        fontsize=8.3, va="bottom", transform=tx.transAxes, color="#444444")
fig.savefig(os.path.join(HERE, "sheet_landmarks.png"), dpi=130)

with open(os.path.join(HERE, "sheet_landmarks.md"), "w") as f:
    f.write("# Observed sheet points (UR16e task, flat sheet at reset)\n\n")
    f.write("| index | where | node (row, col) | robot frame x, y, z [m] |\n|---|---|---|---|\n")
    for i in range(9):
        f.write(f"| {i} | {names[i]} | {ids[i]} {tuple(picks[i])} | {robot[i, 0]:+.3f}, {robot[i, 1]:+.3f}, {robot[i, 2]:+.3f} |\n")
    f.write("\nPairwise distances [cm]:\n\n|   | " + " | ".join(str(j) for j in range(9)) + " |\n|" + "---|" * 10 + "\n")
    for i in range(9):
        f.write(f"| **{i}** | " + " | ".join("" if i == j else f"{dist[i, j]:.1f}" for j in range(9)) + " |\n")
print("wrote", os.path.join(HERE, "sheet_landmarks.png"))
