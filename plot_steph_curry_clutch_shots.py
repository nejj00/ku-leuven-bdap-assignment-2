import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplbasketball import Court

# ── Load data ─────────────────────────────────────────────────────────────────
clutch    = pd.read_csv("clutch_efficiency.csv", sep=" ")
possession = pd.read_csv("possession_per_player.csv", sep=" ")
players   = pd.read_csv("spark_basketball/data/players.csv")

# Get Curry's player_id from players.csv — don't hardcode it
curry_row = players[players["PLAYER_NAME"].str.contains("Stephen Curry", case=False)]
print(curry_row)
CURRY_ID = curry_row["PLAYER_ID"].values[0]
print(f"Stephen Curry player_id: {CURRY_ID}")

# ── Load clutch shot locations ─────────────────────────────────────────────────
# We need the withShotType dataframe which has x_loc, y_loc per shot
# Load the raw moments and events to recompute shot locations for Curry
moments = pd.read_csv("data/moments/0021500548.csv")  # adjust for full dataset
events  = pd.read_csv("data/events/0021500548.csv")

FEET_TO_METERS = 0.3048
moments["x_loc"] = moments["x_loc"] * FEET_TO_METERS
moments["y_loc"] = moments["y_loc"] * FEET_TO_METERS

# ── Identify clutch shot events for Curry ─────────────────────────────────────
# Forward fill SCOREMARGIN
events = events.sort_values(["PERIOD", "PCTIMESTRING"], ascending=[True, False])
events["SCOREMARGIN"] = pd.to_numeric(events["SCOREMARGIN"], errors="coerce")
events["SCOREMARGIN"] = events["SCOREMARGIN"].ffill()

# Parse game clock
events["minutes"] = pd.to_datetime(events["PCTIMESTRING"], format="%M:%S").dt.minute
events["seconds"] = pd.to_datetime(events["PCTIMESTRING"], format="%M:%S").dt.second
events["game_clock_seconds"] = events["minutes"] * 60 + events["seconds"]

curry_clutch = events[
    (events["EVENTMSGTYPE"].isin([1, 2])) &
    (events["PLAYER1_ID"] == CURRY_ID) &
    (events["PERIOD"] >= 4) &
    (events["game_clock_seconds"] <= 300) &
    (events["SCOREMARGIN"].abs() <= 5)
].copy()

print(f"Curry clutch shots found: {len(curry_clutch)}")

# ── Find shot locations from moments using ball height ────────────────────────
ball = moments[moments["player_id"] == -1].copy()
curry_moments = moments[moments["player_id"] == CURRY_ID].copy()

shot_locations = []
for _, shot in curry_clutch.iterrows():
    event_id  = shot["EVENTNUM"]
    quarter   = shot["PERIOD"]
    made      = shot["EVENTMSGTYPE"] == 1
    game_clock_s = shot["game_clock_seconds"]

    # Get ball moments for this event — find first moment ball >= 2m
    ball_event = ball[
        (ball["event_id"] == event_id) &
        (ball["quarter"]  == quarter)
    ].copy()

    if len(ball_event) == 0:
        continue

    # Primary: first moment ball >= 2m (highest game_clock where ball >= 2m)
    above_2m = ball_event[ball_event["radius"] >= 2.0]
    if len(above_2m) > 0:
        shot_clock = above_2m["game_clock"].max()
    else:
        # Fallback: highest ball point for this event
        shot_clock = ball_event.loc[ball_event["radius"].idxmax(), "game_clock"]

    # Find Curry's position closest to shot_clock
    curry_event = curry_moments[
        (curry_moments["event_id"] == event_id) &
        (curry_moments["quarter"]  == quarter)
    ].copy()

    if len(curry_event) == 0:
        continue

    curry_event["clock_diff"] = (curry_event["game_clock"] - shot_clock).abs()
    closest = curry_event.loc[curry_event["clock_diff"].idxmin()]

    shot_locations.append({
        "x_loc"  : closest["x_loc"],
        "y_loc"  : closest["y_loc"],
        "made"   : made,
        "event_id": event_id
    })

shots_df = pd.DataFrame(shot_locations)
print(f"Shot locations resolved: {len(shots_df)}")
print(shots_df)

# ── Flip shots taken toward left basket to right side ────────────────────────
# Court is 94ft * 0.3048 = 28.6512m long
# Half court x = 28.6512 / 2 = 14.3256m
COURT_LENGTH_M = 94 * FEET_TO_METERS   # 28.6512m
HALF_COURT_M   = COURT_LENGTH_M / 2    # 14.3256m

# If player is in the left half (x < half court) they're shooting at left basket
# Flip those shots: new_x = COURT_LENGTH - x, new_y = COURT_LENGTH_Y - y
COURT_WIDTH_M = 50 * FEET_TO_METERS    # 15.24m

shots_df["x_plot"] = shots_df["x_loc"].apply(
    lambda x: COURT_LENGTH_M - x if x < HALF_COURT_M else x
)
shots_df["y_plot"] = shots_df.apply(
    lambda row: COURT_WIDTH_M - row["y_loc"] if row["x_loc"] < HALF_COURT_M else row["y_loc"],
    axis=1
)

# ── Draw court and plot shots ─────────────────────────────────────────────────
# mplbasketball uses feet by default — convert back to feet for the court drawing
# or use the metric option if available
fig, ax = plt.subplots(figsize=(10, 6))

court = Court(court_type="nba", units="ft")
court.draw(ax=ax, half=True)  # draw right half court only

# Convert back to feet for plotting on mplbasketball court
shots_df["x_ft"] = shots_df["x_plot"] / FEET_TO_METERS
shots_df["y_ft"] = shots_df["y_plot"] / FEET_TO_METERS

made   = shots_df[shots_df["made"] == True]
missed = shots_df[shots_df["made"] == False]

ax.scatter(made["x_ft"],   made["y_ft"],   
           c="green", marker="o", s=120, zorder=5, label="Scored")
ax.scatter(missed["x_ft"], missed["y_ft"], 
           c="red",   marker="x", s=120, zorder=5, label="Missed", linewidths=2)

ax.set_title(f"Stephen Curry — Clutch Time Field Goal Attempts", fontsize=13)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.savefig("report/curry_shot_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to report/curry_shot_chart.png")