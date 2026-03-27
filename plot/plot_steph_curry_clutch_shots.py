import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplbasketball import Court
from mplbasketball.utils import transform

shots_df = pd.read_csv("curry_shot_locations.csv", sep=" ")

FEET_TO_METERS = 0.3048
shots_df["x_ft"] = shots_df["x_loc"] / FEET_TO_METERS
shots_df["y_ft"] = shots_df["y_loc"] / FEET_TO_METERS

# Data origin is bottom-left: x ∈ [0, 94], y ∈ [0, 50]
origin     = "bottom-left"
court_type = "nba"

# Transform all shots to right half so all appear toward right basket
x_hr, y_hr = transform(
    shots_df["x_ft"].values.copy(),
    shots_df["y_ft"].values.copy(),
    fr="h",
    to="hr",
    origin=origin,
    court_type=court_type
)

shots_df["x_plot"] = x_hr
shots_df["y_plot"] = y_hr

print("x_plot range:", shots_df["x_plot"].min(), "–", shots_df["x_plot"].max())
print("y_plot range:", shots_df["y_plot"].min(), "–", shots_df["y_plot"].max())

# ── Draw full court ───────────────────────────────────────────────────────────
court = Court(origin=origin, court_type=court_type, units="ft")
fig, ax = court.draw(orientation="h")

made   = shots_df[shots_df["event_type"] == 1]
missed = shots_df[shots_df["event_type"] == 2]

ax.scatter(made["x_plot"],   made["y_plot"],
           c="green", marker="o", s=120, zorder=5, label=f"Scored ({len(made)})")
ax.scatter(missed["x_plot"], missed["y_plot"],
           c="red",   marker="x", s=120, zorder=5, label=f"Missed ({len(missed)})", linewidths=2)

# ax.set_title("Stephen Curry — Clutch Time Field Goal Attempts", fontsize=13)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("report/curry_shot_chart.png", dpi=150, bbox_inches="tight")
plt.show()