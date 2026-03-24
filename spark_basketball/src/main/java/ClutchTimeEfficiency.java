import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class ClutchTimeEfficiency {

    private final Dataset<Row> playerMoments;
    private final Dataset<Row> ballMoments;
    private final Dataset<Row> events;

    private static final double HALF_COURT_M = 94 * 0.3048 / 2; // 14.326m
    private static final double LEFT_BASKET_X_M = 5.25 * 0.3048; // 1.600m
    private static final double BASKET_Y_M = 25.0 * 0.3048; // 7.620m
    private static final double RIGHT_BASKET_X_M = 88.75 * 0.3048; // 27.051m

    private static final double THREE_POINT_RADIUS_M = 6.71;

    private static final int CLUTCH_TIME_SECONDS = 300;
    private static final int CLUTCH_SCORE_MARGIN = 5;

    private static final short EVENT_TYPE_MADE = 1;
    private static final short EVENT_TYPE_MISSED = 2;

    private static final short BALL_HEIGHT_THRESHOLD_M = 2;

    private static final int STEPTH_CURRY_ID = 201939;

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

    public Dataset<Row> getCurryShotLocations() {
        Dataset<Row> clutchShots = identifyClutchShots();
        Dataset<Row> withLocation = joinWithShotLocation(clutchShots);
        Dataset<Row> withShotType = classifyShotType(withLocation);

        return withShotType
                .filter(col("player_id").equalTo(STEPTH_CURRY_ID))
                .select(
                        col("player_id"),
                        col("event_type"),
                        col("shot_type"),
                        col("x_loc"),
                        col("y_loc"),
                        col("dist_to_basket"),
                        col("game_clock_seconds"),
                        col("game_id"));
    }

    private Dataset<Row> identifyClutchShots() {
        WindowSpec fillWindow = Window
                .partitionBy("GAME_ID")
                .orderBy(col("PERIOD").asc(), col("PCTIMESTRING").desc())
                .rowsBetween(Window.unboundedPreceding(), 0);

        return events
                .filter(col("EVENTMSGTYPE").isin(EVENT_TYPE_MADE, EVENT_TYPE_MISSED))
                .withColumn("SCOREMARGIN_FILLED",
                        last(col("SCOREMARGIN"), true).over(fillWindow))
                .withColumn("pc_str", date_format(col("PCTIMESTRING"), "HH:mm"))
                .withColumn("minutes", expr("cast(split(pc_str, ':')[0] as int)"))
                .withColumn("seconds", expr("cast(split(pc_str, ':')[1] as int)"))
                .withColumn("game_clock_seconds", expr("minutes * 60 + seconds"))
                .filter(col("SCOREMARGIN_FILLED").isNotNull())
                .filter(col("PERIOD").geq(4))
                .filter(col("game_clock_seconds").leq(CLUTCH_TIME_SECONDS))
                .filter(abs(col("SCOREMARGIN_FILLED")).leq(CLUTCH_SCORE_MARGIN))
                .select(
                        col("GAME_ID").alias("game_id"),
                        col("EVENTNUM").alias("event_id"),
                        col("PERIOD").alias("quarter"),
                        col("PLAYER1_ID").alias("player_id"),
                        col("EVENTMSGTYPE").alias("event_type"),
                        col("game_clock_seconds"));
    }

    private Dataset<Row> joinWithShotLocation(Dataset<Row> clutchShots) {

        Dataset<Row> playerMomentsAliased = playerMoments
                .select(
                        col("game_id").alias("m_game_id"),
                        col("event_id").alias("m_event_id"),
                        col("player_id").alias("m_player_id"),
                        col("quarter").alias("m_quarter"),
                        col("game_clock").alias("m_game_clock"),
                        col("x_loc"),
                        col("y_loc"));

        WindowSpec eventWindow = Window.partitionBy("game_id", "event_id", "quarter");

        Dataset<Row> shotMoments = ballMoments
                .join(clutchShots,
                        ballMoments.col("game_id").equalTo(clutchShots.col("game_id"))
                                .and(ballMoments.col("event_id").equalTo(clutchShots.col("event_id")))
                                .and(ballMoments.col("quarter").equalTo(clutchShots.col("quarter"))),
                        "inner")
                .select(
                        ballMoments.col("game_id"),
                        ballMoments.col("event_id"),
                        ballMoments.col("quarter"),
                        ballMoments.col("game_clock"),
                        col("radius").alias("ball_height"),
                        clutchShots.col("player_id"),
                        clutchShots.col("event_type"),
                        clutchShots.col("game_clock_seconds"))
                // The shot location is approximated as the location of the ball when it's
                // height hits 2m or is at its max height for the shot (miss or made) event.
                .withColumn("above_2m_clock",
                        max(when(col("ball_height").geq(BALL_HEIGHT_THRESHOLD_M), col("game_clock"))).over(eventWindow))
                .withColumn("max_height", max("ball_height").over(eventWindow))
                .withColumn("max_height_clock",
                        max(when(col("ball_height").equalTo(col("max_height")), col("game_clock"))).over(eventWindow))
                .withColumn("shot_game_clock", coalesce(col("above_2m_clock"), col("max_height_clock")))
                .dropDuplicates("game_id", "event_id", "quarter");

        Dataset<Row> joined = shotMoments.join(playerMomentsAliased,
                shotMoments.col("game_id").equalTo(playerMomentsAliased.col("m_game_id"))
                        .and(shotMoments.col("event_id").equalTo(playerMomentsAliased.col("m_event_id")))
                        .and(shotMoments.col("player_id").equalTo(playerMomentsAliased.col("m_player_id")))
                        .and(shotMoments.col("quarter").equalTo(playerMomentsAliased.col("m_quarter"))),
                "inner");

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
                        col("y_loc"));
    }

    private Dataset<Row> classifyShotType(Dataset<Row> withLocation) {
        return withLocation
                .withColumn("basket_x",
                        when(col("x_loc").lt(HALF_COURT_M), lit(LEFT_BASKET_X_M))
                                .otherwise(lit(RIGHT_BASKET_X_M)))
                .withColumn("basket_y", lit(BASKET_Y_M))
                .withColumn("dist_to_basket",
                        expr("sqrt(pow(x_loc - basket_x, 2) + pow(y_loc - basket_y, 2))"))
                // 2pt if within 6.71m of basket, 3pt if beyond
                .withColumn("shot_type",
                        when(col("dist_to_basket").leq(THREE_POINT_RADIUS_M), lit("2pt"))
                                .otherwise(lit("3pt")))
                .select(
                        col("game_id"),
                        col("player_id"),
                        col("event_type"),
                        col("event_id"),
                        col("shot_type"),
                        col("dist_to_basket"),
                        col("x_loc"),
                        col("y_loc"),
                        col("game_clock_seconds"),
                        col("tracking_game_clock"));
    }

    private Dataset<Row> aggregateEfficiency(Dataset<Row> withShotType) {
        Dataset<Row> agg = withShotType
                .groupBy("player_id", "shot_type")
                .agg(
                        count("*").alias("nb_shots"),
                        sum(when(col("event_type").equalTo(EVENT_TYPE_MADE), 1).otherwise(0)).alias("nb_made"));

        Dataset<Row> pivoted = agg
                .groupBy("player_id")
                .pivot("shot_type", java.util.Arrays.asList("2pt", "3pt"))
                .agg(
                        first("nb_shots").alias("shots"),
                        first("nb_made").alias("made"))
                .na().fill(0); // players with no 2pt or no 3pt attempts get 0

        return pivoted
                .withColumn("2pts_efficiency",
                        when(col("2pt_shots").equalTo(0), lit(0.0))
                                .otherwise(round(col("2pt_made").divide(col("2pt_shots")), 2)))
                .withColumn("3pts_efficiency",
                        when(col("3pt_shots").equalTo(0), lit(0.0))
                                .otherwise(round(col("3pt_made").divide(col("3pt_shots")), 2)))
                .select(
                        col("player_id"),
                        col("2pts_efficiency"),
                        col("2pt_shots").alias("nb_2pts_shots"),
                        col("3pts_efficiency"),
                        col("3pt_shots").alias("nb_3pts_shots"))
                .orderBy(col("2pts_efficiency").desc());
    }
}