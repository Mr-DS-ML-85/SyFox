#!/bin/bash
# ============================================================================
# b77_train_chunked.sh — dedicated bank77 fabric, chunked incremental learn.
# The full 18k-row file (19KB JSON rows, 77-candidate criteria) OOMs the
# 4GB sandbox in one process; chunking keeps each process's peak bounded and
# is exactly how the v3.1.3 giant fabric was trained.
# Usage: bash scripts/b77_train_chunked.sh <start-chunk> <end-chunk-inclusive>
# ============================================================================
set -e
cd /home/z/my-project/syfox

CHUNKS_DIR=data/b77_hier_chunks
MODEL=model-b77-sem
ROWS_PER_CHUNK=1500

if [ ! -f "$CHUNKS_DIR/.split_done" ]; then
    mkdir -p "$CHUNKS_DIR"
    rm -f "$CHUNKS_DIR"/chunk_*.jsonl
    python3 - "$CHUNKS_DIR" "$ROWS_PER_CHUNK" << 'EOF'
import json, sys, os
out_dir, per = sys.argv[1], int(sys.argv[2])
rows = []
with open('data/bank77_hier_train.jsonl') as f:
    for line in f:
        if line.strip():
            rows.append(line)
n_chunks = 0
for i in range(0, len(rows), per):
    with open(f"{out_dir}/chunk_{n_chunks:02d}.jsonl", "w") as f:
        f.writelines(rows[i:i+per])
    n_chunks += 1
open(f"{out_dir}/.split_done", "w").write(str(n_chunks))
print(f"split {len(rows)} rows into {n_chunks} chunks")
EOF
fi

START=${1:-0}
END=${2:-17}
for i in $(seq -f "%02g" "$START" "$END"); do
    f="$CHUNKS_DIR/chunk_$i.jsonl"
    if [ ! -f "$f" ]; then echo "chunk $i missing, done"; break; fi
    echo "=== CHUNK $i $(date +%H:%M:%S) ==="
    ./build/syfox learn --model "$MODEL" --examples "$f" --epochs 3 2>&1 | tail -1
done
echo "=== chunks $START..$END complete ==="
