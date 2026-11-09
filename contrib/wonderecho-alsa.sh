#!/bin/bash
# WonderEcho USB PnP audio: mute mic MONITOR (playback), keep capture.
# Leaving Mic Playback up routes the mic into the speaker → feedback screech.
set -euo pipefail

card=""
for c in /proc/asound/card*/id; do
  id=$(cat "$c" 2>/dev/null || true)
  # Stock Linux name for the WonderEcho / generic USB PnP sound gadget
  if [[ "$id" == "Device" ]]; then
    card=$(basename "$(dirname "$c")" | sed 's/card//')
    break
  fi
done

if [[ -z "$card" ]]; then
  # fallback: first USB audio card that is not HDMI
  while read -r line; do
    if [[ "$line" =~ ^\ *([0-9]+)\ +\[([^\]]+)\].*USB ]]; then
      card="${BASH_REMATCH[1]}"
      break
    fi
  done < <(cat /proc/asound/cards)
fi

if [[ -z "${card}" ]]; then
  echo "[wonderecho-alsa] no USB audio card yet" >&2
  exit 0
fi

echo "[wonderecho-alsa] card=$card id=$(cat /proc/asound/card${card}/id)"

# Speaker at half the old 60% level.
amixer -c "$card" set Speaker 30% unmute >/dev/null || true

# Critical: zero Mic *playback* (sidetone/monitor). Keep capture for ASR.
if amixer -c "$card" set Mic playback 0% mute >/dev/null 2>&1; then
  :
else
  # older amixer: try numeric cset names
  amixer -c "$card" cset name='Mic Playback Volume' 0 >/dev/null 2>&1 || true
  amixer -c "$card" cset name='Mic Playback Switch' off >/dev/null 2>&1 || true
fi

if amixer -c "$card" set Mic capture 80% unmute >/dev/null 2>&1; then
  :
else
  amixer -c "$card" cset name='Mic Capture Volume' 80% >/dev/null 2>&1 || true
  amixer -c "$card" cset name='Mic Capture Switch' on >/dev/null 2>&1 || true
fi

amixer -c "$card" sget Mic 2>/dev/null | sed 's/^/[wonderecho-alsa] /' || true
