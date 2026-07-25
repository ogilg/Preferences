#!/usr/bin/env bash
# Pull steering checkpoints off the pod every few minutes so a pod loss can't
# destroy generated rows. Exits once single-task generation reaches its full
# 9000 rows. Usage: bash scripts/reviewer_followups/sync_pod_checkpoints.sh <ip> <port>
set -uo pipefail

IP="$1"
PORT="$2"
KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/workspace/Preferences/experiments/reviewer_followups/user_context_persona/checkpoints"
LOCAL_DIR="experiments/reviewer_followups/user_context_persona/checkpoints"
TARGET=9000

mkdir -p "$LOCAL_DIR"

pull() {
  for f in sadist_user_context_contrastive.jsonl sadist_user_context_single_task.jsonl; do
    # Write to a temp file first so an interrupted transfer can't truncate a good local copy.
    if ssh -o ConnectTimeout=15 "root@$IP" -p "$PORT" -i "$KEY" "cat $REMOTE_DIR/$f" > "$LOCAL_DIR/$f.tmp" 2>/dev/null; then
      if [ -s "$LOCAL_DIR/$f.tmp" ]; then
        mv "$LOCAL_DIR/$f.tmp" "$LOCAL_DIR/$f"
      else
        rm -f "$LOCAL_DIR/$f.tmp"
      fi
    else
      rm -f "$LOCAL_DIR/$f.tmp"
    fi
  done
}

while true; do
  pull
  c=$(wc -l < "$LOCAL_DIR/sadist_user_context_contrastive.jsonl" 2>/dev/null | tr -d ' ')
  s=$(wc -l < "$LOCAL_DIR/sadist_user_context_single_task.jsonl" 2>/dev/null | tr -d ' ')
  echo "synced: contrastive=${c:-0}/4500 single_task=${s:-0}/$TARGET"
  if [ "${s:-0}" -ge "$TARGET" ]; then
    echo "single-task generation complete; final sync done"
    exit 0
  fi
  sleep 180
done
