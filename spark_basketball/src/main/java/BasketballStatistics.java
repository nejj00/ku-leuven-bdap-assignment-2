import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class BasketballStatistics {

    private static void distanceTravelled(String fileName) {

        SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("Distance Travelled");

        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
    }

    public static void main(String[] args) {

        // if (args.length == 0) {
        //     System.out.println("No files provided.");
        //     System.exit(0);
        // }

        
        SparkSession spark = SparkSession
        .builder()
        .appName("Java Spark SQL basic example")
        .config("spark.some.config.option", "some-value")
        .getOrCreate();

        Dataset<Row> df = spark.read().csv("/home/neji/ku-leuven-bdap-assignment-2/spark_basketball/data/teams.csv");

        // Displays the content of the DataFrame to stdout
        df.show();
    }
}
