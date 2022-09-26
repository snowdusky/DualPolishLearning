#!/usr/bin/env bash
set -x
OFFSET=$RANDOM
for fold in 1 2 3 4 5; do
    python tools/dataset/split_voc.py --percent 50 --seed ${fold} --data-dir /data0/sunyuxuan/VOC/VOCdevkit --seed-offset ${OFFSET}
done
