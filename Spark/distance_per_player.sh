#!/bin/bash

MASTER="yarn"
DATA_DIR="/data/nba_movement_data"
OUTPUT_DIR="/user/r1035493/output"

spark-submit \
    --class BasketballStatistics \
    --master $MASTER \
    target/*.jar \
    distance \
    $DATA_DIR \
    $OUTPUT_DIR