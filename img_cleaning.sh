#!/bin/bash
cd /Users/zoesun/Documents/final-project/data/dataset_1/train/paper || exit 1

i=1
for file in paper*.jpg; do
    mv -- "$file" "$(printf 'paper_%03d.jpg' "$i")"
    ((i++))
done