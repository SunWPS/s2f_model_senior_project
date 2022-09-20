#!/bin/bash

DIR_NAME=$1
START=$2

id=$START
echo "###### Start rename file" $(date)

for file in $(ls -lR og_data/${DIR_NAME})
do
    if grep -e ".jpg$" -e ".jpeg$" -e ".png$" <<< ${file}; then
        full_path=$(find og_data/${DIR_NAME} -type f -name $file)
        
        for f in $full_path
        do 
            target=$(printf %06d $id)
            mv $f "data/$target.jpg"
            echo "$f,$target.jpg" >> renameTo_${DIR_NAME}.csv
            id=$(( $id + 1 ))
        done
    fi
done

echo "###### Happy ending" $(date)
