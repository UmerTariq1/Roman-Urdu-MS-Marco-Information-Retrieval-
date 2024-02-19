#!/bin/bash

batch_size=64
num_workers=8
num_beams=8
max_seq_len=256
model_name="ai4bharat/indictrans2-en-indic-1B"
output_dir="/local/umerbutt/thesis/data/mmarco/output/"
input_file="/local/umerbutt/thesis/data/mmarco/collections/english_collection_part_ab.tsv"

export CUDA_VISIBLE_DEVICES=2; python translate_en_ur_2.py \
    --batch_size "$batch_size" \
    --num_workers "$num_workers" \
    --num_beams "$num_beams" \
    --max_seq_len "$max_seq_len" \
    --model_name "$model_name" \
    --output_dir "$output_dir" \
    --input_file "$input_file"