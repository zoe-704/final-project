#!/bin/bash
cd /Users/zoesun/Documents/final-project/test/rock || exit 1

i=1
for file in rock*.jpg; do
    mv -- "$file" "$(printf 'rock_%03d.jpg' "$i")"
    ((i++))
done