#!/usr/bin/env bash
# download the data set from https://www.kaggle.com/c/dogs-vs-cats/data
# extract it into /all and replace the path here:
cd /Users/xlewang/Downloads/temp/all/train
mkdir -p ../sample/train/cats
mkdir -p ../sample/train/dogs
mkdir -p ../sample/valid/cats
mkdir -p ../sample/valid/dogs

cp cat.1??.jpg ../sample/train/cats
cp dog.1??.jpg ../sample/train/dogs
cp cat.2??.jpg ../sample/valid/cats
cp dog.2??.jpg ../sample/valid/dogs

echo 'done'

