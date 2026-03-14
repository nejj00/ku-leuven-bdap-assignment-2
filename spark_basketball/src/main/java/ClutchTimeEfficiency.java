import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class ClutchTimeEfficiency {

    private final Dataset<Row> playerMoments;
    private final Dataset<Row> ballMoments;
    private final Dataset<Row> events;

    // After converting to meters, half court is at x = 94 * 0.3048 / 2 = 14.326m
    private static final double HALF_COURT_M     = 94 * 0.3048 / 2;  // 14.326m
    private static final double LEFT_BASKET_X_M  = 5.25  * 0.3048;   // 1.600m
    private static final double BASKET_Y_M  = 25.0  * 0.3048;   // 7.620m
    private static final double RIGHT_BASKET_X_M = 88.75 * 0.3048;   // 27.051m

// 3-point line radius from the assignment: 6.71 meters
private static final double THREE_POINT_RADIUS_M = 6.71;

    public ClutchTimeEfficiency(Dataset<Row> playerMoments, Dataset<Row> ballMoments, Dataset<Row> events) {
        this.playerMoments = playerMoments;
        this.ballMoments = ballMoments;
        this.events = events;
    }

    public Dataset<Row> compute() {
        Dataset<Row> clutchShots = identifyClutchShots();
        clutchShots = clutchShots.persist();
        Dataset<Row> withLocation = joinWithShotLocation(clutchShots);
        Dataset<Row> withShotType = classifyShotType(withLocation);
        Dataset<Row> result = aggregateEfficiency(withShotType);
        clutchShots.unpersist();

        return result;
    }

    // Step 1: Filter events to clutch time shot attempts only
    private Dataset<Row> identifyClutchShots() {
        WindowSpec fillWindow = Window
                .partitionBy("GAME_ID")
                .orderBy(col("PERIOD").asc(), col("PCTIMESTRING").desc())
                .rowsBetween(Window.unboundedPreceding(), 0);

        return events
                // filter to shots first — reduces rows before the expensive window fill
                .filter(col("EVENTMSGTYPE").isin(1, 2))
                .withColumn("SCOREMARGIN_FILLED",
                    last(col("SCOREMARGIN"), true).over(fillWindow)
                )
                .withColumn("pc_str",             date_format(col("PCTIMESTRING"), "HH:mm"))
                .withColumn("minutes",            expr("cast(split(pc_str, ':')[0] as int)"))
                .withColumn("seconds",            expr("cast(split(pc_str, ':')[1] as int)"))
                .withColumn("game_clock_seconds", expr("minutes * 60 + seconds"))
                .filter(col("SCOREMARGIN_FILLED").isNotNull())
                .filter(col("PERIOD").geq(4))
                .filter(col("game_clock_seconds").leq(300))
                .filter(abs(col("SCOREMARGIN_FILLED")).leq(5))
                .select(
                    col("GAME_ID").alias("game_id"),
                    col("EVENTNUM").alias("event_id"),
                    col("PERIOD").alias("quarter"),
                    col("PLAYER1_ID").alias("player_id"),
                    col("EVENTMSGTYPE").alias("event_type"),
                    col("game_clock_seconds")
                );
    }

    // Step 2: Find the player's position at the moment the shot was taken.
    // The assignment warns that pbp and tracking timestamps may not be perfectly
    // synced, so we join on event_id and then pick the moment whose game_clock
    // is closest to the shot's game_clock_seconds among all moments for that event.
    private Dataset<Row> joinWithShotLocation(Dataset<Row> clutchShots) {

        // Get ball moments (player_id == -1), radius is the ball's height in meters
        Dataset<Row> ballMomentsAliased = ballMoments
                .select(
                    col("game_id").alias("b_game_id"),
                    col("event_id").alias("b_event_id"),
                    col("quarter").alias("b_quarter"),
                    col("game_clock").alias("b_game_clock"),
                    col("radius").alias("ball_height")
                );

        // Get player moments for position lookup
        Dataset<Row> playerMomentsAliased = playerMoments
                .select(
                    col("game_id").alias("m_game_id"),
                    col("event_id").alias("m_event_id"),
                    col("player_id").alias("m_player_id"),
                    col("quarter").alias("m_quarter"),
                    col("game_clock").alias("m_game_clock"),
                    col("x_loc"),
                    col("y_loc")
                );

        // Find the first moment in each event where the ball is >= 2m off the ground
        // game_clock counts DOWN so ordering descending = chronological order
        // The LAST game_clock value (i.e. highest game_clock) where ball >= 2m
        // is the first moment the ball rose to shot height — that's the release point
        WindowSpec shotMomentWindow = Window
                .partitionBy("b_game_id", "b_event_id", "b_quarter")
                .orderBy(col("b_game_clock").desc());

        Dataset<Row> shotMoments = ballMomentsAliased
                .filter(col("ball_height").geq(2.0))
                .withColumn("rn", row_number().over(shotMomentWindow))
                .filter(col("rn").equalTo(1))  // first moment ball reached 2m (highest game_clock = earliest)
                .select(
                    col("b_game_id"),
                    col("b_event_id"),
                    col("b_quarter"),
                    col("b_game_clock").alias("shot_game_clock"),
                    col("ball_height")
                );

        // Join clutch shots with the shot moment to get the game_clock of release
        Dataset<Row> shotsWithClock = clutchShots.join(shotMoments,
                clutchShots.col("game_id").equalTo(shotMoments.col("b_game_id"))
                .and(clutchShots.col("event_id").equalTo(shotMoments.col("b_event_id")))
                .and(clutchShots.col("quarter").equalTo(shotMoments.col("b_quarter"))),
                "inner"
        );

        // Now join with player moments at the exact shot game_clock
        Dataset<Row> joined = shotsWithClock.join(playerMomentsAliased,
                shotsWithClock.col("game_id").equalTo(playerMomentsAliased.col("m_game_id"))
                .and(shotsWithClock.col("event_id").equalTo(playerMomentsAliased.col("m_event_id")))
                .and(shotsWithClock.col("player_id").equalTo(playerMomentsAliased.col("m_player_id")))
                .and(shotsWithClock.col("quarter").equalTo(playerMomentsAliased.col("m_quarter"))),
                "inner"
        );

        // Among all player moments for this event, pick the one closest to shot_game_clock
        WindowSpec pickClosest = Window
                .partitionBy("game_id", "event_id", "player_id", "quarter")
                .orderBy(abs(col("m_game_clock").minus(col("shot_game_clock"))).asc());

        return joined
                .withColumn("rn", row_number().over(pickClosest))
                .filter(col("rn").equalTo(1))
                .select(
                    col("game_id"),
                    col("player_id"),
                    col("event_id"),
                    col("quarter"),
                    col("event_type"),
                    col("game_clock_seconds"),
                    col("shot_game_clock").alias("tracking_game_clock"),
                    col("ball_height"),
                    col("x_loc"),
                    col("y_loc")
                );
    }

    // Step 3: Determine which basket the player is shooting at based on their
    // x position, compute distance to that basket, and classify as 2pt or 3pt.
    private Dataset<Row> classifyShotType(Dataset<Row> withLocation) {
        return withLocation
                // Player on left half → shooting at left basket, right half → right basket
                .withColumn("basket_x",
                    when(col("x_loc").lt(HALF_COURT_M), lit(LEFT_BASKET_X_M))
                    .otherwise(lit(RIGHT_BASKET_X_M))
                )
                .withColumn("basket_y", lit(BASKET_Y_M))
                .withColumn("dist_to_basket",
                    expr("sqrt(pow(x_loc - basket_x, 2) + pow(y_loc - basket_y, 2))")
                )
                // 2pt if within 6.71m of basket, 3pt if beyond
                .withColumn("shot_type",
                    when(col("dist_to_basket").leq(THREE_POINT_RADIUS_M), lit("2pt"))
                    .otherwise(lit("3pt"))
                )
                .select(
                    col("game_id"),
                    col("player_id"),
                    col("event_type"),      // 1=made, 2=missed
                    col("event_id"),
                    col("shot_type"),       // "2pt" or "3pt"
                    col("dist_to_basket"),
                    col("x_loc"),
                    col("y_loc"),
                    col("game_clock_seconds"),
                    col("tracking_game_clock")
                );
    }

    private Dataset<Row> aggregateEfficiency(Dataset<Row> withShotType) {
        // Count total attempts and made shots per player per shot type
        Dataset<Row> agg = withShotType
                .groupBy("player_id", "shot_type")
                .agg(
                    count("*").alias("nb_shots"),
                    sum(when(col("event_type").equalTo(1), 1).otherwise(0)).alias("nb_made")
                );

        // Pivot to get one row per player with 2pt and 3pt columns side by side
        Dataset<Row> pivoted = agg
                .groupBy("player_id")
                .pivot("shot_type", java.util.Arrays.asList("2pt", "3pt"))
                .agg(
                    first("nb_shots").alias("shots"),
                    first("nb_made").alias("made")
                )
                .na().fill(0);  // players with no 2pt or no 3pt attempts get 0

        return pivoted
                .withColumn("2pts_efficiency",
                    when(col("2pt_shots").equalTo(0), lit(0.0))
                    .otherwise(round(col("2pt_made").divide(col("2pt_shots")), 2))
                )
                .withColumn("3pts_efficiency",
                    when(col("3pt_shots").equalTo(0), lit(0.0))
                    .otherwise(round(col("3pt_made").divide(col("3pt_shots")), 2))
                )
                .select(
                    col("player_id"),
                    col("2pts_efficiency"),
                    col("2pt_shots").alias("nb_2pts_shots"),
                    col("3pts_efficiency"),
                    col("3pt_shots").alias("nb_3pts_shots")
                )
                .orderBy(col("2pts_efficiency").desc());
    }
}