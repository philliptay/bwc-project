IN_DATA_FILE="/n/fs/vai-bwc-scr/annotations/categories/category_annotations.txt"
IN_DATA_DIR="/n/fs/vai-bwc-scr/videos"
OUT_DATA_DIR="/n/fs/vai-bwc-scr/action_detection/frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

while read file
do
  video="$file"
  
  OIFS=$IFS
  IFS=$'.'
  read -a strarr <<< "$video"
  video=${strarr[0]}
  IFS=$OIFS

  video_name=${video##*/}
  video_name=${video_name// /_}
  #video_name=${video_name::-4}
  
  echo ${video_name}

  in_video=${IN_DATA_DIR}/${video}
  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  if [[ ! -d "${out_video_dir}" ]]; then
      mkdir -p "${out_video_dir}"

      out_name="${out_video_dir}/${video_name}_%06d.jpg"

      ffmpeg -i "${in_video}.mp4" -r 30 -q:v 1 "${out_name}"
  fi
done < "${IN_DATA_FILE}"
IFS=$SAVEIFS
