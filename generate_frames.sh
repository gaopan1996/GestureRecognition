#!/bin/bash
i=1
for filename in "ASL/"*; do
    echo $filename
    python gesture_detect.py -f $filename
    i=$((i + 1))
done
