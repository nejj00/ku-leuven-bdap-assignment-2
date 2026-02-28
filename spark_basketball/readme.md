# Spark Project – Setup & Run Cheat Sheet

This project is part of BDAP Assignment 2 (Spark – Basketball Data Analysis).

Tested with:
- Java 11
- Hadoop 3.4.3
- Spark 3.5.8
- Maven 3+

---

# 1. Prerequisites

## Java

Check version:
```bash
java -version
```

Set environment variables:
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
export PATH=$JAVA_HOME/bin:$PATH
```

---

## Hadoop (for cluster / HDFS)

Download:
https://hadoop.apache.org/releases.html

Install:
```bash
tar -xvzf hadoop-3.4.3.tar.gz
sudo mv hadoop-3.4.3 /opt/hadoop
```

Set environment variables:
```bash
export HADOOP_HOME=/opt/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
```

Test:
```bash
hadoop version
```

---

## Spark

Download:
https://spark.apache.org/downloads.html

Install:
```bash
tar -xvzf spark-3.5.8-bin-hadoop3.tgz
sudo mv spark-3.5.8-bin-hadoop3 /opt/spark
```

Set environment variables:
```bash
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin
```

Test:
```bash
spark-submit --version
```

---

## Maven

Install:
```bash
sudo apt install maven
```

Test:
```bash
mvn -version
```

---

## Env variables

You can set the env variable to `nano ~/.bashrc` as:

```bash
# Hadoop env variables
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME=/usr/local/hadoop
export HADOOP_INSTALL=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME

export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Spaark env variables
export SPARK_HOME=/usr/local/spark
export PATH=$PATH:$SPARK_HOME/bin
export SPARK_LOCAL_IP=127.0.0.1
```


# 2. Project Structure

Example layout:

```
basketball-stats/
│
├── pom.xml
├── src/
│   └── main/
│       └── java/
│           └── BasketballStatistics.java
│
├── target/
│   └── basketball-stats-1.0.jar
│
└── input/
```

---

# 3. Example pom.xml

```xml
<project>
    <modelVersion>4.0.0</modelVersion>

    <groupId>DTAI</groupId>
    <artifactId>basketball-stats</artifactId>
    <version>1.0</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-sql_2.12</artifactId>
            <version>3.5.8</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>
</project>
```

Note: `scope=provided` is required for cluster execution.

---

# 4. Build the Project

From project root:

```bash
mvn clean package
```

Generated JAR:
```
target/basketball-stats-1.0.jar
```

---

# 5. Run Locally

Single-core execution:

```bash
spark-submit \
  --class BasketballStatistics \
  --master local[1] \
  target/basketball-stats-1.0.jar
```

With input/output arguments:

```bash
spark-submit \
  --class BasketballStatistics \
  --master local[1] \
  target/basketball-stats-1.0.jar \
  input_path output_path
```

---

# 6. Run on Cluster (YARN)

```bash
spark-submit \
  --class BasketballStatistics \
  --master yarn \
  target/basketball-stats-1.0.jar \
  /data/nba_movement_data \
  output_folder
```

Check jobs:
```bash
hadoop job -list
```

Kill job:
```bash
hadoop job -kill <job_id>
```

---

# 7. SparkSession Template (Java)

```java
SparkSession spark = SparkSession.builder()
    .appName("Basketball Statistics")
    .getOrCreate();
```

SparkSQL must be used for assignment queries.

---

# 8. Typical Workflow

1. Develop and test on single-game sample dataset.
2. Verify correctness locally.
3. Optimize transformations (avoid unnecessary shuffles).
4. Run on full DFS dataset.
5. Generate required CSV outputs:
   - distance_per_player.csv
   - possession_per_player.csv
   - clutch_efficiency.csv
6. Ensure formatting, sorting, and rounding match assignment requirements.

---

# 9. Common Issues

Native Hadoop warning:
```
Unable to load native-hadoop library...
```
Safe to ignore.

Class not found:
- Verify `--class` name
- Ensure JAR exists in `target/`
- Run `mvn package` again

Output folder exists:
```bash
rm -r output_folder
```

---

# 10. Before Submission

- Correct filenames
- Runnable `.sh` scripts included
- SparkSQL used
- Output sorted and rounded correctly
- No hardcoded local paths
- Runs on departmental machines
- Source code included in `src/`