import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

possession = pd.read_csv("possession_per_player.csv", sep=" ")

print(f"Players: {len(possession)}")
print(possession["percentage_ball_possession"].describe())

possession["possession_fraction"] = possession["percentage_ball_possession"] / 100

fig, ax = plt.subplots(figsize=(9, 5))

counts, bins, _ = ax.hist(
    possession["possession_fraction"],
    bins=10,
    color="#2563eb",
    edgecolor="white",
    linewidth=0.5
)

mean_val = possession["possession_fraction"].mean()
ax.axvline(mean_val, color="#dc2626", linestyle="--", linewidth=1.5,
           label=f"Mean: {mean_val:.4f}")

# x-axis: double the number of ticks by adding midpoints between bin edges
bin_edges    = bins
bin_midpoints = (bins[:-1] + bins[1:]) / 2
x_ticks = np.sort(np.concatenate([bin_edges, bin_midpoints]))
ax.set_xticks(x_ticks)
ax.set_xticklabels([f"{t:.3f}" for t in x_ticks], rotation=45, ha="right", fontsize=8)

ax.set_xlabel("Fraction of time in ball possession (normalized by total time on court)", fontsize=12)
ax.set_ylabel("Number of players", fontsize=12)
ax.set_title("Distribution of ball possession time across players", fontsize=13)
ax.legend()

plt.tight_layout()
plt.savefig("report/possession_histogram.png", dpi=150, bbox_inches="tight")
plt.show()