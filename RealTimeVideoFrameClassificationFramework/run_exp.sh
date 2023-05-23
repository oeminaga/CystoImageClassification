#!/bin/bash
FILES="/home/eminaga/PCIe/DevelopmentSet_CystoNet/UpperTract_New/*"
for f in $FILES
do
    python main.py --video_source  "$f"
done