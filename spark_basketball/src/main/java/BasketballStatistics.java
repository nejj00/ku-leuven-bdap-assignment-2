import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import static org.apache.spark.sql.functions.*;

public class BasketballStatistics {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Usage: BasketballStatistics <path_to_data_dir>");
            System.exit(0);
        }

        String dataDir = args[0];
        // Ensure no trailing slash
        if (dataDir.endsWith("/")) {
            dataDir = dataDir.substring(0, dataDir.length() - 1);
        }

        SparkSession spark = SparkSession.builder()
                .appName("Basketball Statistics")
                .getOrCreate();

        // Load CSVs
        Dataset<Row> games = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/games.csv");

        Dataset<Row> players = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/players.csv");

        Dataset<Row> teams = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/teams.csv");

        Dataset<Row> minutesPlayed = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/minutes_played.csv");

        // Events and moments are directories containing one CSV per game
        Dataset<Row> events = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/events/*.csv");

        Dataset<Row> moments = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/moments/*.csv");

        // Show first 5 rows of each
        System.out.println("=== games ===");
        games.show(5);

        System.out.println("=== players ===");
        players.show(5);

        System.out.println("=== teams ===");
        teams.show(5);

        System.out.println("=== minutes_played ===");
        minutesPlayed.show(5);

        System.out.println("=== events ===");
        events.show(5);

        // 3.1 Total distance 
        System.out.println("=== moments ===");
        moments.show(100);

        // FIlter out ball moments
        Dataset<Row> players_moments = moments.filter(col("player_id").notEqual(-1));
        System.out.println("=== players_moments ===");
        players_moments.show(100);

        // Deduplicate moments (duplicates have happened becaause more than one event can happen at the same timestamp)
        Dataset<Row> players_moments_deduped = players_moments.dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .sort(col("game_id"), col("player_id"), col("quarter"), col("game_clock").desc());
                
        System.out.println("=== players_moments_deduped ===");
        players_moments_deduped.show(100);

        WindowSpec window = Window
                .partitionBy("game_id", "player_id", "quarter")
                .orderBy(col("quarter").asc(), col("game_clock").desc());

        Dataset<Row> players_moments_deduped_with_next = players_moments_deduped
                .withColumn("x_next", lead("x_loc", 1).over(window))
                .withColumn("y_next", lead("y_loc", 1).over(window));

        System.out.println("=== players_moments_deduped_with_next ===");
        players_moments_deduped_with_next.show(100);

        // Calculate distance between current and next moment
        Dataset<Row> players_moments_distance = players_moments_deduped_with_next.withColumn(
                "dist_m",
                expr("(sqrt(pow(x_loc - x_next, 2) + pow(y_loc - y_next, 2))) * 0.3048")); // Convert feet to meters
        
        System.out.println("=== players_moments_distance ===");
        players_moments_distance.show(100);

        // Calculate total distance per player
        Dataset<Row> total_distance_per_player = players_moments_distance.groupBy("player_id")
                .agg(sum("dist_m").alias("total_distance_m"))
                .orderBy(col("total_distance_m").desc());

        System.out.println("=== total_distance_per_player ===");
        total_distance_per_player.show(100);

        // Total minutes played per player
        Dataset<Row> total_seconds = minutesPlayed.groupBy("player_id")
                .agg(sum("sec").alias("total_seconds"))
                .orderBy(col("total_seconds").desc());

        Dataset<Row> total_minutes = total_seconds.withColumn("total_minutes", col("total_seconds").divide(60))
                .orderBy(col("total_minutes").desc());

        System.out.println("=== total_minutes ===");
        total_minutes.show(100);

        Dataset<Row> distance_and_minutes = total_distance_per_player.join(total_minutes, "player_id")
                .select("player_id", "total_distance_m", "total_minutes");
        
        System.out.println("=== distance_and_minutes ===");
        distance_and_minutes.show(100);

        
        Dataset<Row> total_distance_per_quater = distance_and_minutes.withColumn(
                "distance_per_quarter_m",
                expr("total_distance_m * 12 / total_minutes"))
                .orderBy(col("distance_per_quarter_m").desc()); // 12 minutes per quarter

        System.out.println("=== total_distance_per_quater ===");
        total_distance_per_quater.show(100);


        spark.stop();
    }
}