#!/usr/bin/env python3
"""Milestone-5 gate: the OMP parallel settle must be BIT-IDENTICAL to the
sequential settle. Compares the full bench JSON blocks (routing, calibration,
honesty, guardrail) of the sequential and --threads 2 runs on the same model
and hidden test."""
import json, sys

seq = json.load(open('build/omp-seq-full.json'))
par = json.load(open('build/omp-par-full.json'))
for block in ('routing', 'calibration', 'honesty', 'guardrail'):
    if seq.get(block) != par.get(block):
        sys.exit(f'FAIL: {block} differs between sequential and OMP settle:\n'
                 f'  seq: {seq.get(block)}\n  par: {par.get(block)}')
print('ok  OMP settle bit-identical to sequential '
      f"(choice acc {seq['routing']['choice_accuracy']})")
