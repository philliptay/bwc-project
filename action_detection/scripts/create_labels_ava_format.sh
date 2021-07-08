#!/bin/bash

DATA_DIR="/n/fs/vai-bwc-scr/action_detection/frames"
OUT_FILE="/n/fs/vai-bwc-scr/action_detection/annotations/bwc_ava_predictions.csv"

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

vid_num=0
for video in $(ls -A1 -U ${DATA_DIR}/*)
do
  video_name=${video##*/}
  video_dir=${OUT_DATA_DIR}/${video_name}/
  frame_id=0
  for frame in $(ls -A1 -U ${video_dir}/*)
  do
      line=$video_name" "$vid_num" "$frame_id" "${video_dir}/${frame}" \"\""
      echo $line >> OUT_FILE
      ((frame_id++))
  done
  
  ((vid_num++))
done
IFS=$SAVEIFS
