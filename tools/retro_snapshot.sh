#!/usr/bin/env bash
set -e
cd "$(git rev-parse --show-toplevel)"

git fetch --tags origin
git add -A
git commit -m "Auto retro snapshot: $(date '+%Y-%m-%d %H:%M:%S %Z')" || true

SNAP="retro-$(date '+%Y%m%d-%H%M')"
git tag -f "$SNAP"
git push origin "refs/tags/$SNAP"

mapfile -t TAGS < <(git tag -l 'retro-*' --sort=-creatordate)
if [ "${#TAGS[@]}" -gt 15 ]; then
  for ((i=15;i<${#TAGS[@]};i++)); do
    git tag -d "${TAGS[$i]}" || true
    git push --delete origin "${TAGS[$i]}" || true
  done
fi
