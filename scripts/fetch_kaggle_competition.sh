#!/usr/bin/env bash
set -e

competition_name="$1"
nickname="${2:-$competition_name}"

cd /workspaces/Kaggle-tools
cd data

kaggle competitions download "$competition_name" -w -q -o

if [competition_name != "$nickname"]; then
    mv -f "${competition_name}.zip" "${nickname}.zip"
fi
mkdir -p "$nickname"
unzip -o "${nickname}.zip" -d "$nickname"
rm -f "${nickname}.zip"

echo "Done, data is at location: data/$nickname"
