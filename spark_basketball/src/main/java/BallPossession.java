import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class BallPossession {

    private final Dataset<Row> moments;
    private final Dataset<Row> minutesPlayed;

    private final double SECONDS_PER_MOMENT = 1.0 / 25.0;

    public BallPossession(Dataset<Row> moments, Dataset<Row> minutesPlayed) {
        this.moments = moments;
        this.minutesPlayed = minutesPlayed;
    }

    public Dataset<Row> compute() {
        // Separate ball and player rows
        Dataset<Row> ball = moments.filter(col("player_id").equalTo(-1))
                .select(
                    col("game_id").alias("ball_game_id"),
                    col("quarter").alias("ball_quarter"),
                    col("game_clock").alias("ball_game_clock"),
                    col("x_loc").alias("ball_x"),
                    col("y_loc").alias("ball_y")
                );

        Dataset<Row> players = moments.filter(col("player_id").notEqual(-1));

        // Join every player row with the ball position at the same moment
        Dataset<Row> withBall = players.join(ball, 
                players.col("game_id").equalTo(ball.col("ball_game_id"))
                .and(players.col("quarter").equalTo(ball.col("ball_quarter")))
                .and(players.col("game_clock").equalTo(ball.col("ball_game_clock"))),
                "inner"
        );

        System.out.println("=== withBall (each player moment joined with ball position) ===");
        withBall.show(100);

        // Compute distance from each player to the ball at each moment
        Dataset<Row> withDist = withBall.withColumn("dist_to_ball",
                expr("sqrt(pow(x_loc - ball_x, 2) + pow(y_loc - ball_y, 2))")
        );

        // Flag each player as "near ball" if within 0.5 meters
        Dataset<Row> withNearFlag = withDist.withColumn("near_ball",
                when(col("dist_to_ball").leq(0.5), 1).otherwise(0)
        );

        System.out.println("=== withNearFlag (each player moment with near ball flag) ===");
        withNearFlag.show(100);

        // Per moment, count how many players are near the ball
        WindowSpec momentWindow = Window.partitionBy("game_id", "quarter", "game_clock");

        Dataset<Row> withNearCount = withNearFlag.withColumn("players_near_ball",
                sum("near_ball").over(momentWindow)
        );

        // A player has possession if:
        // (1) they are within 0.5m of the ball AND
        // (2) they are the ONLY player within 0.5m
        Dataset<Row> possession = withNearCount.withColumn("has_possession",
                when(
                    col("near_ball").equalTo(1).and(col("players_near_ball").equalTo(1)),
                    true
                ).otherwise(false)
        );

        // Keep only the columns we care about
        // Each row is one player at one moment — has_possession=true means they own the ball
        Dataset<Row> possessionMoments = possession.filter(col("has_possession").equalTo(true)).select(
                col("game_id"),
                col("quarter"),
                col("game_clock"),
                col("player_id"),
                col("team_id"),
                col("x_loc"),
                col("y_loc"),
                col("dist_to_ball"),
                col("players_near_ball"),
                col("has_possession")
        // ).sort(col("players_near_ball").desc());
        ).sort(col("game_id"), col("quarter"), col("game_clock").desc(), col("player_id"));

        possessionMoments.show(100);

        WindowSpec window = Window
                .partitionBy("game_id", "player_id", "quarter")
                .orderBy(col("game_clock").desc());

        Dataset<Row> withDistancePossessionMoments = possessionMoments
                .withColumn("x_next", lead("x_loc", 1).over(window))
                .withColumn("y_next", lead("y_loc", 1).over(window))
                .withColumn("dist_m", expr(
                        "sqrt(pow(x_loc - x_next, 2) + pow(y_loc - y_next, 2))"
                ));
        
        Dataset<Row> distanceTravelledWhileInPossession = withDistancePossessionMoments
                .groupBy("player_id")
                .agg(
                    sum("dist_m").alias("total_distance_with_ball_possession_m"),   
                    (count("*").multiply(SECONDS_PER_MOMENT)).alias("time_ball_possession_s")
                );

        distanceTravelledWhileInPossession.show(100);

        Dataset<Row> totalTimeOnCourt = minutesPlayed
            .filter(col("game_id").equalTo(21500548)) //TODO Remove this for production case 
            .groupBy("player_id")
            .agg(sum("sec").alias("total_time_on_court_s"));
        
        
        Dataset<Row> result = distanceTravelledWhileInPossession
            .join(totalTimeOnCourt, "player_id", "left")
            .withColumn("percentage_ball_possession",
                round(
                    col("time_ball_possession_s")
                        .divide(col("total_time_on_court_s"))
                        .multiply(100),
                    2
                )
            )
            .select(
                col("player_id"),
                col("total_distance_with_ball_possession_m"),
                col("time_ball_possession_s"),
                col("total_time_on_court_s"),
                col("percentage_ball_possession")
            );

        result.show(100);

        return result;
    }
}
