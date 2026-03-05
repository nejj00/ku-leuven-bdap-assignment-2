import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class DistanceTravelled {

    private final Dataset<Row> moments;
    private final Dataset<Row> minutesPlayed;

    public DistanceTravelled(Dataset<Row> moments, Dataset<Row> minutesPlayed) {
        this.moments = moments;
        this.minutesPlayed = minutesPlayed;
    }

    public Dataset<Row> compute() {
        Dataset<Row> playerMoments = filterAndDeduplicate();
        Dataset<Row> withDistances = computeDistances(playerMoments);
        Dataset<Row> totalDistance = aggregateTotalDistance(withDistances);
        Dataset<Row> totalMinutes  = aggregateTotalMinutes();
        return computeDistancePerQuarter(totalDistance, totalMinutes);
    }

    // Filter out ball rows and deduplicate (same position logged under multiple events)
    private Dataset<Row> filterAndDeduplicate() {
        return moments
                .filter(col("player_id").notEqual(-1))
                .sort(col("game_id"), col("player_id"), col("quarter"), col("game_clock").desc());
    }

    // Add x_next/y_next columns and compute Euclidean distance in meters
    private Dataset<Row> computeDistances(Dataset<Row> playerMoments) {
        // Partition by quarter so lead() does not connect across quarter boundaries
        WindowSpec window = Window
                .partitionBy("game_id", "player_id", "quarter")
                .orderBy(col("game_clock").desc());

        return playerMoments
                .withColumn("x_next", lead("x_loc", 1).over(window))
                .withColumn("y_next", lead("y_loc", 1).over(window))
                .withColumn("dist_m", expr(
                        "sqrt(pow(x_loc - x_next, 2) + pow(y_loc - y_next, 2))"
                ));
    }

    // Sum distances per player across all games
    private Dataset<Row> aggregateTotalDistance(Dataset<Row> withDistances) {
        return withDistances
                .groupBy("player_id")
                .agg(sum("dist_m").alias("total_distance_m"));
    }

    // Sum seconds played per player across all games, convert to minutes
    private Dataset<Row> aggregateTotalMinutes() {
        return minutesPlayed
                .groupBy("player_id")
                .agg(sum("sec").alias("total_seconds"))
                .withColumn("total_minutes", col("total_seconds").divide(60));
    }

    // Join distance and minutes, apply normalization formula
    private Dataset<Row> computeDistancePerQuarter(Dataset<Row> totalDistance, Dataset<Row> totalMinutes) {
        return totalDistance
                .join(totalMinutes, "player_id")
                // Formula: TotalDistance * 12 / minutes_played  (12 min per quarter)
                .withColumn("distance_per_quarter_m",
                        round(expr("total_distance_m * 12 / total_minutes"), 2))
                .select("player_id", "distance_per_quarter_m")
                .orderBy(col("distance_per_quarter_m").desc());
    }
}