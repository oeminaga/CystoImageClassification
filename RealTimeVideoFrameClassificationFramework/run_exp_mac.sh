#!/bin/bash
source ~/miniconda/bin/activate base
FILES="../examples/*"
for f in $FILES
do
    python main.py --video_source  "$f" -c
done