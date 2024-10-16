#!/bin/bash

current_dir=$(pwd)

parent_dir=$(basename "$(dirname "$current_dir")")
current_dir_name=$(basename "$current_dir")

last_two_dirs="$parent_dir/$current_dir_name"
expected_dirs="KU_IDEC/Day2"

if [ "$last_two_dirs" != "$expected_dirs" ]; then
  echo "Error: This script must be run from the KU_IDEC/Day2 directory."
  exit 1
fi

cat split_files/data_alexnet_vgg16.tar.gz.part* > data_alexnet_vgg16.tar.gz
tar -xvzf data_alexnet_vgg16.tar.gz

rm -rf ./split_files
rm -rf ./data_alexnet_vgg16.tar.gz