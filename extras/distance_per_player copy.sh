#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <local|cluster>"
    exit 1
fi

MODE=$1

case $MODE in
    local)
        MASTER="local[1]"
        DATA_DIR="data"
        OUTPUT_DIR="output"
        ;;
    cluster)
        MASTER="yarn"
        DATA_DIR="/data/nba_movement_data"
        OUTPUT_DIR="/user/r1035493/output"
        ;;
    *)
        echo "Unknown mode '$MODE'. Use 'local' or 'cluster'."
        exit 1
        ;;
esac

spark-submit \
    --class BasketballStatistics \
    --master $MASTER \
    target/*.jar \
    distance \
    $DATA_DIR \
    $OUTPUT_DIR