#!/bin/bash

curl -L -o animals10.zip https://www.kaggle.com/api/v1/datasets/download/alessiocorrado99/animals10 && \
unzip -q animals10.zip -d data && \
rm animals10.zip

echo "Dataset downloaded and extracted to 'data' directory."