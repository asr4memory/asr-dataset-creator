#!/bin/bash

# Run the complete asr dataset creator workflow
echo "COPYING FILES TO WORKING DIRECTORIES"
cp ./data/general/_input_vtt/* ./data/vtt_to_txt/_input/
cp ./data/general/_input_vtt/* ./data/vtt_anonymization/_input_vtt/
cp ./data/general/_input_wav/* ./data/dataset_creator/_input_wav/

echo "RUNNING VTT_TO_TXT SCRIPT"
python3 vtt-to-txt.py

echo "COPYING FILES TO WORKING DIRECTORIES"
cp ./data/vtt_to_txt/_output/* ./data/ner_workflow/_input/
rm -r ./data/vtt_to_txt/_output/*
rm -r ./data/vtt_to_txt/_input/*

echo "RUNNING NER_WORKFLOW SCRIPT"
python3 ner-workflow.py

echo "COPYING FILES TO WORKING DIRECTORIES"
cp ./data/ner_workflow/_output/*.json ./data/vtt_anonymization/_input_json/
rm -r ./data/ner_workflow/_output/*
rm -r ./data/ner_workflow/_input/*

echo "RUNNING VTT_ANONYMIZATION SCRIPT"
python3 anonymize-vtt.py

echo "COPYING FILES TO WORKING DIRECTORIES"
cp ./data/vtt_anonymization/_output/* ./data/dataset_creator/_input_vtt/
rm -r ./data/vtt_anonymization/_output/*
rm -r ./data/vtt_anonymization/_input_vtt/*
rm -r ./data/vtt_anonymization/_input_json/*

echo "RUNNING DATASET_CREATOR SCRIPT"
python3 asr-dataset-creator.py

echo "COPYING FILES TO WORKING DIRECTORIES"
cp -r ./data/dataset_creator/_output/* ./data/dataset_merger/_input/
rm -r ./data/dataset_creator/_input_vtt/*
rm -r ./data/dataset_creator/_input_wav/*

echo "RUNNING DATASET_MERGER SCRIPT"
python3 asr-dataset-merger.py
cp -r ./data/dataset_merger/_output/* ./data/general/_output/
rm -r ./data/dataset_merger/_output/*
rm -r ./data/dataset_merger/_input/*
