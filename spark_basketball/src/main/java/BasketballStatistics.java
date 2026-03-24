import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import static org.apache.spark.sql.functions.*;

public class BasketballStatistics {

    enum Task {
        DISTANCE,
        POSSESSION,
        CLUTCH,
        CURRY;

        public static Task fromString(String s) {
            switch (s.toLowerCase().trim()) {
                case "distance":
                    return DISTANCE;
                case "possession":
                    return POSSESSION;
                case "clutch":
                    return CLUTCH;
                case "curry":
                    return CURRY;
                default:
                    throw new IllegalArgumentException(
                            "Unknown task '" + s
                                    + "'. Valid options: distance, possession, clutch");
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println(
                    "Usage: BasketballStatistics <task> <path_to_data_dir> <path_to_output_dir>");
            System.out.println("Tasks: distance | possession | clutch | curry");
            System.exit(0);
        }

        Task task = Task.fromString(args[0]);
        String dataDir = args[1].endsWith("/") ? args[1].substring(0, args[1].length() - 1) : args[1];
        String outputDir = args[2].endsWith("/") ? args[2].substring(0, args[2].length() - 1) : args[2];

        System.out.println("Running task: " + task);

        SparkSession spark = SparkSession.builder()
                .appName("Basketball Statistics — " + task)
                .getOrCreate();

        FileSystem fs = FileSystem.get(spark.sparkContext().hadoopConfiguration());

        final double FEET_TO_METERS = 0.3048;

        switch (task) {
            case DISTANCE:
                runDistance(spark, fs, dataDir, outputDir, FEET_TO_METERS);
                break;
            case POSSESSION:
                runPossession(spark, fs, dataDir, outputDir, FEET_TO_METERS);
                break;
            case CLUTCH:
                runClutch(spark, fs, dataDir, outputDir, FEET_TO_METERS);
                break;
            case CURRY:
                runCurry(spark, fs, dataDir, outputDir, FEET_TO_METERS);
                break;
        }

        spark.stop();
    }

    private static void runDistance(SparkSession spark, FileSystem fs,
            String dataDir, String outputDir,
            double feetToMeters) throws Exception {
        Dataset<Row> minutesPlayed = loadCSV(spark, dataDir + "/minutes_played.csv");

        Dataset<Row> playerMoments = loadMoments(spark, dataDir, feetToMeters)
                .filter(col("player_id").notEqual(-1))
                .dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .cache();

        Dataset<Row> result = new DistanceTravelled(playerMoments, minutesPlayed).compute();
        saveAsCSV(result, outputDir, "distance_per_player.csv", fs);

        playerMoments.unpersist();
    }

    private static void runPossession(SparkSession spark, FileSystem fs,
            String dataDir, String outputDir,
            double feetToMeters) throws Exception {
        Dataset<Row> minutesPlayed = loadCSV(spark, dataDir + "/minutes_played.csv");

        Dataset<Row> momentsInMeters = loadMoments(spark, dataDir, feetToMeters);

        Dataset<Row> playerMoments = momentsInMeters
                .filter(col("player_id").notEqual(-1))
                .dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .cache();

        Dataset<Row> ballMoments = momentsInMeters
                .filter(col("player_id").equalTo(-1))
                .dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                .cache();

        Dataset<Row> result = new BallPossession(playerMoments, ballMoments, minutesPlayed).compute();
        saveAsCSV(result, outputDir, "possession_per_player.csv", fs);

        playerMoments.unpersist();
        ballMoments.unpersist();
    }

    private static void runClutch(SparkSession spark, FileSystem fs,
            String dataDir, String outputDir,
            double feetToMeters) throws Exception {
        Dataset<Row> events = loadCSV(spark, dataDir + "/events/*.csv");

        Dataset<Row> momentsInMeters = loadMoments(spark, dataDir, feetToMeters);

        Dataset<Row> playerMoments = momentsInMeters
                .filter(col("player_id").notEqual(-1))
                .cache();

        Dataset<Row> ballMoments = momentsInMeters
                .filter(col("player_id").equalTo(-1))
                .cache();

        Dataset<Row> result = new ClutchTimeEfficiency(playerMoments, ballMoments, events).compute();
        saveAsCSV(result, outputDir, "clutch_efficiency.csv", fs);

        playerMoments.unpersist();
        ballMoments.unpersist();
    }

    private static void runCurry(SparkSession spark, FileSystem fs,
            String dataDir, String outputDir,
            double feetToMeters) throws Exception {

        Dataset<Row> events = loadCSV(spark, dataDir + "/events/*.csv");

        Dataset<Row> momentsInMeters = loadMoments(spark, dataDir, feetToMeters);

        Dataset<Row> playerMoments = momentsInMeters
                .filter(col("player_id").notEqual(-1))
                .cache();

        Dataset<Row> ballMoments = momentsInMeters
                .filter(col("player_id").equalTo(-1))
                .cache();

        Dataset<Row> result = new ClutchTimeEfficiency(playerMoments, ballMoments, events)
                .getCurryShotLocations();
        saveAsCSV(result, outputDir, "curry_shot_locations.csv", fs);

        playerMoments.unpersist();
        ballMoments.unpersist();
    }

    private static Dataset<Row> loadCSV(SparkSession spark, String path) {
        return spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(path);
    }

    private static Dataset<Row> loadMoments(SparkSession spark, String dataDir, double feetToMeters) {
        return loadCSV(spark, dataDir + "/moments/*.csv")
                .withColumn("x_loc", col("x_loc").multiply(feetToMeters))
                .withColumn("y_loc", col("y_loc").multiply(feetToMeters))
                .withColumn("radius", col("radius").multiply(feetToMeters));
    }

    private static void saveAsCSV(Dataset<Row> df, String outputDir,
            String filename, FileSystem fs) throws Exception {
        String tmpPath = outputDir + "/tmp_" + filename;
        String finalPath = outputDir + "/" + filename;

        df.coalesce(1)
                .write()
                .option("header", "false")
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