#!/bin/bash

experiment_dir="experiments"
code_dir="${experiment_dir}/three_stage"
config="option/three_stage.json"

if [ ! -d "$experiment_dir" ]
then
  mkdir "$experiment_dir"
else
  echo "experiment dir exists"
fi

if [ ! -d "${code_dir}" ]
then
  mkdir "${code_dir}"
fi

cp -r "dataset" "${code_dir}"
cp -r "model" "${code_dir}"
cp -r "network" "${code_dir}"
cp -r "option" "${code_dir}"
cp -r "train" "${code_dir}"
cp -r "utils" "${code_dir}"
cp -r "validate" "${code_dir}"
cp "train.sh" "${code_dir}"

if [ ! -d "${code_dir}/logs" ]
then
  mkdir "${code_dir}/logs"
fi

cd ${code_dir}
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=$PWD:$PYTHONPATH
python train/train.py --config ${config}