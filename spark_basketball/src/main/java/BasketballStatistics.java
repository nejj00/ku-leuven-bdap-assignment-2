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

        // TODO Make the loading also into a separate class maybe.
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

        final double FEET_TO_METERS = 0.3048;
        
        moments = moments.dropDuplicates("game_id", "player_id", "quarter", "game_clock")
                         .withColumn("x_loc",  col("x_loc").multiply(FEET_TO_METERS))
                         .withColumn("y_loc",  col("y_loc").multiply(FEET_TO_METERS))
                         .withColumn("radius", col("radius").multiply(FEET_TO_METERS));

        System.out.println("=== moments ===");
        moments.show(100);

        // 3.1 Total distance 
        DistanceTravelled distanceTravelled = new DistanceTravelled(moments, minutesPlayed);
        Dataset<Row> distancePerQuarter = distanceTravelled.compute();
        distancePerQuarter.show(100);



        spark.stop();
    }
}