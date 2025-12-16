#!/usr/bin/env bash
set -e

DEST=$1

# Replace .venv directory
if [ -e "$DEST/.venv" ]; then
  rm -rf "$DEST/.venv"
fi
mv /workspaces/.venv "$DEST"

# Replace files
for f in pyproject.toml uv.lock; do
  if [ -e "$DEST/$f" ]; then
    rm -f "$DEST/$f"
  fi
  mv "/workspaces/$f" "$DEST"
done

echo "export PYTHONPATH=$DEST:$PYTHONPATH" >> ~/.bashrc
