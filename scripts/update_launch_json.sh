#!/bin/bash
export DEBUG_HOST=$(squeue --me --name=interactive --states=R -h -O NodeList | xargs)

# 如果 DEBUG_HOST 為空，設置默認值
if [ -z "$DEBUG_HOST" ]; then
    export DEBUG_HOST="localhost"
fi

echo $DEBUG_HOST

# python scripts/clean_launch_json.py

# # 將 DEBUG_HOST 的值寫入 .vscode/launch.json
# jq --arg host "$DEBUG_HOST" \
#     '(.configurations[] | select(.name == "Rank 0 attach") | .connect.host) = $host' \
#     .vscode/launch_cleaned.json > .vscode/launch.json.tmp && mv .vscode/launch.json.tmp .vscode/launch.json