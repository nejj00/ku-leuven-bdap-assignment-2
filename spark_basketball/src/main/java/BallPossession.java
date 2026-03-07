import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class BallPossession {

    private final Dataset<Row> moments;
    private final Dataset<Row> minutesPlayed;

    private static final double SECONDS_PER_MOMENT = 1.0 / 25.0;
    private static final double POSSESSION_RADIUS_M = 0.5;

    public BallPossession(Dataset<Row> moments, Dataset<Row> minutesPlayed) {
        this.moments = moments;
        this.minutesPlayed = minutesPlayed;
    }

    public Dataset<Row> compute() {
        Dataset<Row> possessionMoments  = getPossessionMoments();
        Dataset<Row> possessionStats    = aggregatePossessionStats(possessionMoments);
        Dataset<Row> totalTimeOnCourt   = aggregateTotalTimeOnCourt();
        return buildResult(possessionStats, totalTimeOnCourt);
    }

    // Join each player moment with the ball position at the same timestamp,
    // then flag whether that player has sole possession of the ball.
    private Dataset<Row> getPossessionMoments() {
        Dataset<Row> ball    = extractBall();
        Dataset<Row> players = moments.filter(col("player_id").notEqual(-1));

        Dataset<Row> withBall = joinPlayersWithBall(players, ball);
        Dataset<Row> withDist = computeDistanceToBall(withBall);
        Dataset<Row> withFlag = flagNearBall(withDist);
        Dataset<Row> withPossession = flagPossession(withFlag);

        return withPossession
                .filter(col("has_possession").equalTo(true))
                .select(
                    col("game_id"),
                    col("quarter"),
                    col("game_clock"),
                    col("player_id"),
                    col("team_id"),
                    col("x_loc"),
                    col("y_loc")
                )
                .sort(col("game_id"), col("quarter"), col("game_clock").desc(), col("player_id"));
    }

    // Extract ball rows with aliased columns to avoid ambiguity in the join
    private Dataset<Row> extractBall() {
        return moments.filter(col("player_id").equalTo(-1))
                .select(
                    col("game_id").alias("ball_game_id"),
                    col("quarter").alias("ball_quarter"),
                    col("game_clock").alias("ball_game_clock"),
                    col("x_loc").alias("ball_x"),
                    col("y_loc").alias("ball_y")
                );
    }

    // Join each player row with the ball row at the exact same moment
    private Dataset<Row> joinPlayersWithBall(Dataset<Row> players, Dataset<Row> ball) {
        return players.join(ball,
                players.col("game_id").equalTo(ball.col("ball_game_id"))
                .and(players.col("quarter").equalTo(ball.col("ball_quarter")))
                .and(players.col("game_clock").equalTo(ball.col("ball_game_clock"))),
                "inner"
        );
    }

    // Compute Euclidean distance from each player to the ball (already in meters)
    private Dataset<Row> computeDistanceToBall(Dataset<Row> withBall) {
        return withBall.withColumn("dist_to_ball",
                expr("sqrt(pow(x_loc - ball_x, 2) + pow(y_loc - ball_y, 2))")
        );
    }

    // Flag each player row as near the ball (1) or not (0)
    private Dataset<Row> flagNearBall(Dataset<Row> withDist) {
        return withDist.withColumn("near_ball",
                when(col("dist_to_ball").leq(POSSESSION_RADIUS_M), 1).otherwise(0)
        );
    }

    // Count how many players are near the ball per moment, then flag sole possession
    private Dataset<Row> flagPossession(Dataset<Row> withNearFlag) {
        WindowSpec momentWindow = Window.partitionBy("game_id", "quarter", "game_clock");

        return withNearFlag
                .withColumn("players_near_ball", sum("near_ball").over(momentWindow))
                .withColumn("has_possession",
                    when(
                        col("near_ball").equalTo(1).and(col("players_near_ball").equalTo(1)),
                        true
                    ).otherwise(false)
                );
    }

    // For each player, compute distance traveled and time spent while in possession
    private Dataset<Row> aggregatePossessionStats(Dataset<Row> possessionMoments) {
        WindowSpec window = Window
                .partitionBy("game_id", "player_id", "quarter")
                .orderBy(col("game_clock").desc());

        return possessionMoments
                .withColumn("x_next", lead("x_loc", 1).over(window))
                .withColumn("y_next", lead("y_loc", 1).over(window))
                .withColumn("dist_m", expr(
                    "sqrt(pow(x_loc - x_next, 2) + pow(y_loc - y_next, 2))"
                ))
                .groupBy("player_id")
                .agg(
                    sum("dist_m").alias("total_distance_with_ball_possession_m"),
                    count("*").multiply(SECONDS_PER_MOMENT).alias("time_ball_possession_s")
                );
    }

    // Sum seconds played per player across all games
    private Dataset<Row> aggregateTotalTimeOnCourt() {
        return minutesPlayed
                .groupBy("player_id")
                .agg(sum("sec").alias("total_time_on_court_s"));
    }

    // Join possession stats with court time and compute possession percentage
    private Dataset<Row> buildResult(Dataset<Row> possessionStats, Dataset<Row> totalTimeOnCourt) {
        return possessionStats
                .join(totalTimeOnCourt, "player_id", "left")
                .withColumn("percentage_ball_possession",
                    round(col("time_ball_possession_s").divide(col("total_time_on_court_s")).multiply(100), 2)
                )
                .select(
                    col("player_id"),
                    col("total_distance_with_ball_possession_m"),
                    col("time_ball_possession_s"),
                    col("total_time_on_court_s"),
                    col("percentage_ball_possession")
                );
    }
}