import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class BallPossession {

    private final Dataset<Row> playerMoments;
    private final Dataset<Row> ballMoments;
    private final Dataset<Row> minutesPlayed;

    private static final double SECONDS_PER_MOMENT = 1.0 / 25.0;
    private static final double POSSESSION_RADIUS_M = 0.5;

        

    public BallPossession(Dataset<Row> playerMoments, Dataset<Row> ballMoments, Dataset<Row> minutesPlayed) {
        this.playerMoments = playerMoments;
        this.ballMoments = ballMoments;
        this.minutesPlayed = minutesPlayed;
    }

    public Dataset<Row> compute() {
        Dataset<Row> possessionMoments = getPossessionMoments().persist();
        Dataset<Row> possessionStats = aggregatePossessionStats(possessionMoments);
        Dataset<Row> totalTimeOnCourt = aggregateTotalTimeOnCourt();
        Dataset<Row> result = buildResult(possessionStats, totalTimeOnCourt);
        possessionMoments.unpersist();
        return result;
    }


    private Dataset<Row> getPossessionMoments() {
        Dataset<Row> ball = extractBall();

        return joinPlayersWithBall(playerMoments, ball)
                // Euclidean distance between player and ball
                .withColumn("dist_to_ball",
                        expr("sqrt(pow(x_loc - ball_x, 2) + pow(y_loc - ball_y, 2))"))
                .withColumn("near_ball",
                        when(col("dist_to_ball").leq(POSSESSION_RADIUS_M), 1).otherwise(0))
                // Count the number of players near the ball at each moment
                .withColumn("players_near_ball",
                        sum("near_ball").over(Window.partitionBy("game_id", "quarter", "game_clock")))
                // Keep only moments where thre is only one person near the ball
                .filter(
                        col("near_ball").equalTo(1).and(col("players_near_ball").equalTo(1)))
                .select(
                        col("game_id"),
                        col("quarter"),
                        col("game_clock"),
                        col("player_id"),
                        col("team_id"),
                        col("x_loc"),
                        col("y_loc"));
    }

    private Dataset<Row> extractBall() {
        return ballMoments
                .select(
                        col("game_id").alias("ball_game_id"),
                        col("quarter").alias("ball_quarter"),
                        col("game_clock").alias("ball_game_clock"),
                        col("x_loc").alias("ball_x"),
                        col("y_loc").alias("ball_y"));
    }

    private Dataset<Row> joinPlayersWithBall(Dataset<Row> players, Dataset<Row> ball) {
        return players.join(ball,
                players.col("game_id").equalTo(ball.col("ball_game_id"))
                        .and(players.col("quarter").equalTo(ball.col("ball_quarter")))
                        .and(players.col("game_clock").equalTo(ball.col("ball_game_clock"))),
                "inner");
    }

    private Dataset<Row> aggregatePossessionStats(Dataset<Row> possessionMoments) {
        WindowSpec window = Window
                .partitionBy("game_id", "player_id", "quarter")
                .orderBy(col("game_clock").desc());

        return possessionMoments
                .withColumn("x_next", lead("x_loc", 1).over(window))
                .withColumn("y_next", lead("y_loc", 1).over(window))
                .withColumn("clock_next", lead("game_clock", 1).over(window))
                // As with the DistanceTravelled class we avoid counting large jumps from non subsequent moments
                .withColumn("dist_m", expr(
                        "case when x_next is not null and (game_clock - clock_next) <= 0.5 " +
                                "then sqrt(pow(x_loc - x_next, 2) + pow(y_loc - y_next, 2)) " +
                                "else 0 end"))
                .groupBy("player_id")
                .agg(
                        sum("dist_m").alias("total_distance_with_ball_possession_m"),
                        // Number of moments withh possession is multipled by the seconds per moment
                        count("*").multiply(SECONDS_PER_MOMENT).alias("time_ball_possession_s"));
    }

    private Dataset<Row> aggregateTotalTimeOnCourt() {
        return minutesPlayed
                .groupBy("player_id")
                .agg(sum("sec").alias("total_time_on_court_s"));
    }

    private Dataset<Row> buildResult(Dataset<Row> possessionStats, Dataset<Row> totalTimeOnCourt) {
        return possessionStats
                .join(totalTimeOnCourt, "player_id", "left")
                .withColumn("percentage_ball_possession",
                        round(col("time_ball_possession_s").divide(col("total_time_on_court_s")).multiply(100), 2))
                .select(
                        col("player_id"),
                        col("total_distance_with_ball_possession_m"),
                        col("time_ball_possession_s"),
                        col("total_time_on_court_s"),
                        col("percentage_ball_possession"));
    }
}