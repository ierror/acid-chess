#!/usr/bin/env bash

for json in */*/*.json
do
 echo "Processing $json";
 jq "del(.imageData)" $json | jq ". + {imageData: (null)}"  > tmp.json && mv tmp.json $json;
done
