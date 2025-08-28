#!/usr/bin/env bash
set -e
ROOT="$(cd "$(git rev-parse --show-toplevel)"; pwd)"
LINE="0 22 * * * cd $ROOT && bash tools/retro_snapshot.sh >/tmp/retro_snapshot.log 2>&1"
( crontab -l 2>/dev/null | grep -v "tools/retro_snapshot.sh" ; echo "$LINE" ) | crontab -
