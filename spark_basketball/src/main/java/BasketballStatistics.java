import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import static org.apache.spark.sql.functions.*;

public class BasketballStatistics {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: BasketballStatistics <path_to_data_dir> <path_to_output_dir>");
            System.exit(0);
        }

        String dataDir   = args[0].endsWith("/") ? args[0].substring(0, args[0].length() - 1) : args[0];
        String outputDir = args[1].endsWith("/") ? args[1].substring(0, args[1].length() - 1) : args[1];

        SparkSession spark = SparkSession.builder()
                .appName("Basketball Statistics")
                .getOrCreate();

        FileSystem fs = FileSystem.get(spark.sparkContext().hadoopConfiguration());

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

        // ── Step 1: Convert all positional columns to meters once ──────────
        // All downstream tasks use meters — do this conversion a single time
        final double FEET_TO_METERS = 0.3048;

        Dataset<Row> momentsInMeters = moments
                .withColumn("x_loc",  col("x_loc").multiply(FEET_TO_METERS))
                .withColumn("y_loc",  col("y_loc").multiply(FEET_TO_METERS))
                .withColumn("radius", col("radius").multiply(FEET_TO_METERS));
        
        // ── Split into ball and player moments once ────────────────────────
        // Cached since multiple classes read from these
        // Deduped player moments — used by DistanceTravelled and BallPossession
        Dataset<Row> playerMoments = momentsInMeters
                .filter(col("player_id").notEqual(-1))
                .cache();
        
        Dataset<Row> playerMomentsDeduped = playerMoments
                .dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .cache();

        Dataset<Row> ballMoments = momentsInMeters
                .filter(col("player_id").equalTo(-1))
                .cache();
        
        Dataset<Row> ballMomentsDeduped = ballMoments
                .dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .cache();


        // ── 3.2.1 Distance per player ──────────────────────────────────────
        Dataset<Row> distancePerQuarter = new DistanceTravelled(playerMomentsDeduped, minutesPlayed).compute();
        saveAsCSV(distancePerQuarter, outputDir, "distance_per_player.csv", fs);

        // ── 3.2.2 Ball possession ──────────────────────────────────────────
        Dataset<Row> possession = new BallPossession(playerMomentsDeduped, ballMomentsDeduped, minutesPlayed).compute();
        saveAsCSV(possession, outputDir, "possession_per_player.csv", fs);

        // ── 3.2.3 Clutch time efficiency ───────────────────────────────────
        Dataset<Row> clutchEfficiency = new ClutchTimeEfficiency(playerMoments, ballMoments, events).compute();
        saveAsCSV(clutchEfficiency, outputDir, "clutch_efficiency.csv", fs);

        momentsInMeters.unpersist();

        spark.stop();
    }

    // Write to a temp directory, then rename the part file to the exact required filename
    private static void saveAsCSV(Dataset<Row> df, String outputDir,
                                   String filename, FileSystem fs) throws Exception {
        String tmpPath   = outputDir + "/tmp_" + filename;
        String finalPath = outputDir + "/" + filename;

        df.coalesce(1)
          .write()
          .option("header", "true")
          .option("delimiter", " ")
          .mode("overwrite")
          .csv(tmpPath);

        FileStatus[] files = fs.listStatus(new Path(tmpPath));
        Path partFile = null;
        for (FileStatus f : files) {
            if (f.getPath().getName().startsWith("part-")) {
                partFile = f.getPath();
                break;
            }
        }

        if (partFile == null) {
            throw new RuntimeException("No part file found in " + tmpPath);
        }

        fs.delete(new Path(finalPath), false);
        fs.rename(partFile, new Path(finalPath));
        fs.delete(new Path(tmpPath), true);

        System.out.println("Saved: " + finalPath);
    }
}