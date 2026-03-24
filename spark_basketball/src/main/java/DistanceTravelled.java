import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class DistanceTravelled {

    private final Dataset<Row> playerMoments;
    private final Dataset<Row> minutesPlayed;

    public DistanceTravelled(Dataset<Row> playerMoments, Dataset<Row> minutesPlayed) {
        this.playerMoments = playerMoments;
        this.minutesPlayed = minutesPlayed;
    }

    public Dataset<Row> compute() {
        Dataset<Row> withDistances = computeDistances();
        Dataset<Row> totalDistance = aggregateTotalDistance(withDistances);
        Dataset<Row> totalMinutes = aggregateTotalMinutes();
        return computeDistancePerQuarter(totalDistance, totalMinutes);
    }

    private Dataset<Row> computeDistances() {
        WindowSpec window = Window
                .partitionBy("game_id", "player_id", "quarter")
                .orderBy(col("game_clock").desc());

        return playerMoments
                .withColumn("x_next", lead("x_loc", 1).over(window))
                .withColumn("y_next", lead("y_loc", 1).over(window))
                .withColumn("clock_next", lead("game_clock", 1).over(window))
                // TODO Explain in the report that 0.5 is picked after doing some EDA on the
                // distribution of time differences between subsequent moments. This helps to
                // avoid counting large jumps when a player is substituted or data is miissing.
                // Calculate distance from current moment to the next one if the next moment is
                // within 0.5 seconds, otherwise set to 0.
                // This helps to avoid counting large jumps when a player is substituted or data
                // is miissing.
                .withColumn("dist_m", expr(
                        "case when x_next is not null and (game_clock - clock_next) <= 0.5 " +
                                "then sqrt(pow(x_loc - x_next, 2) + pow(y_loc - y_next, 2)) " +
                                "else 0 end"));
    }

    private Dataset<Row> aggregateTotalDistance(Dataset<Row> withDistances) {
        return withDistances
                .groupBy("player_id")
                .agg(sum("dist_m").alias("total_distance_m"));
    }

    private Dataset<Row> aggregateTotalMinutes() {
        return minutesPlayed
                .groupBy("player_id")
                .agg(sum("sec").alias("total_seconds"))
                .withColumn("total_minutes", col("total_seconds").divide(60));
    }

    private Dataset<Row> computeDistancePerQuarter(Dataset<Row> totalDistance, Dataset<Row> totalMinutes) {
        return totalDistance
                .join(totalMinutes, "player_id")
                .withColumn("distance_per_quarter_m",
                        round(expr("total_distance_m * 12 / total_minutes"), 2))
                .select("player_id", "distance_per_quarter_m")
                .orderBy(col("distance_per_quarter_m").desc());
    }
}