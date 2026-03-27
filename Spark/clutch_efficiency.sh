#!/bin/bash

# Build the project first
mvn clean package

# Optional: stop if build fails
if [ $? -ne 0 ]; then
    echo "Maven build failed. Exiting."
    exit 1
fi

MASTER="yarn"
DATA_DIR="/data/nba_movement_data"
OUTPUT_DIR="/user/r1035493/output"

spark-submit \
    --class BasketballStatistics \
    --master $MASTER \
    target/*.jar \
    clutch \
    $DATA_DIR \
    $OUTPUT_DIR