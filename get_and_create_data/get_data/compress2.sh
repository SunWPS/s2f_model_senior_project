#!/bin/bash

DIR_NAME=$1
START=$2

id=$START
echo "###### Start rename file" $(date)

for file in $(ls -lr --format=single-column og_data/${DIR_NAME})
do
    if grep -e ".jpg" -e ".png" <<< ${file}; then
        target=$(printf %06d $id)
        mv "og_data/$DIR_NAME/$file" "data/$target.jpg"
        echo "og_data/$DIR_NAME/$file,$target.jpg" >> renameTo_${DIR_NAME}.csv
        id=$(( $id + 1 ))
    fi
done

echo "###### Happy ending" $(date)


