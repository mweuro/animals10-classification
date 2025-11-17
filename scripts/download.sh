#!/bin/bash

curl -L -o animals10.zip https://www.kaggle.com/api/v1/datasets/download/alessiocorrado99/animals10 && \
unzip -q animals10.zip -d . && \
rm animals10.zip
rm translate.py
echo "Dataset downloaded!"