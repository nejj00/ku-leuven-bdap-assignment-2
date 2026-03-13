import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

public class BasketballStatistics {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: BasketballStatistics <path_to_data_dir> <path_to_output_dir>");
            System.exit(0);
        }

        String dataDir   = args[0].endsWith("/") ? args[0].substring(0, args[0].length() - 1) : args[0];
        String outputDir = args[1].endsWith("/") ? args[1].substring(0, args[1].length() - 1) : args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Basketball Statistics")
                .getOrCreate();

        // ── Load CSVs ──────────────────────────────────────────────────────
        Dataset<Row> minutesPlayed = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/minutes_played.csv");

        Dataset<Row> events = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/events/*.csv");

        Dataset<Row> moments = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(dataDir + "/moments/*.csv");

        // ── Preprocess moments ─────────────────────────────────────────────
        final double FEET_TO_METERS = 0.3048;

        Dataset<Row> dedupedMoments = moments
                .dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .withColumn("x_loc",  col("x_loc").multiply(FEET_TO_METERS))
                .withColumn("y_loc",  col("y_loc").multiply(FEET_TO_METERS))
                .withColumn("radius", col("radius").multiply(FEET_TO_METERS))
                .sort(col("game_id"), col("quarter"), col("game_clock").desc(), col("player_id"));

        // Clutch moments are not deduped — we need all ball height readings
        Dataset<Row> momentsInMeters = moments
                .withColumn("x_loc",  col("x_loc").multiply(FEET_TO_METERS))
                .withColumn("y_loc",  col("y_loc").multiply(FEET_TO_METERS))
                .withColumn("radius", col("radius").multiply(FEET_TO_METERS));

        // ── 3.2.1 Distance per player ──────────────────────────────────────
        Dataset<Row> distancePerQuarter = new DistanceTravelled(dedupedMoments, minutesPlayed).compute();
        distancePerQuarter.show(100);
        saveAsCSV(distancePerQuarter, outputDir + "/distance_per_player");

        // ── 3.2.2 Ball possession ──────────────────────────────────────────
        Dataset<Row> possession = new BallPossession(dedupedMoments, minutesPlayed).compute();
        possession.show(100);
        saveAsCSV(possession, outputDir + "/possession_per_player");

        // ── 3.2.3 Clutch time efficiency ───────────────────────────────────
        Dataset<Row> clutchEfficiency = new ClutchTimeEfficiency(momentsInMeters, events).compute();
        clutchEfficiency.show(100);
        saveAsCSV(clutchEfficiency, outputDir + "/clutch_efficiency");

        spark.stop();
    }

    // Merge all partitions into a single CSV file in the given directory
    private static void saveAsCSV(Dataset<Row> df, String path) {
        df.coalesce(1)
          .write()
          .option("header", "true")
          .option("delimiter", " ")
          .mode("overwrite")
          .csv(path);
    }
}