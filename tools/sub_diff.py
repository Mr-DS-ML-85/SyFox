#!/usr/bin/env python3
"""Structural diff of two substrate.bin bodies: node masses vs lane weights."""
import struct, sys

def parse(path):
    b = open(path, 'rb').read()
    i = b.find(bytes([0x34, 0x4D, 0x45, 0x53]))          # SEM4 magic (LE bytes)
    body = b[:i] if i >= 0 else b
    off = 0
    (n,) = struct.unpack_from('<I', body, off); off += 4
    nodes = []
    for _ in range(n):
        (ln,) = struct.unpack_from('<I', body, off); off += 4
        concept = body[off:off+ln].decode('utf-8', 'replace'); off += ln
        (mass,) = struct.unpack_from('<f', body, off); off += 4
        nodes.append((concept, mass))
    (lanes,) = struct.unpack_from('<I', body, off); off += 4
    L = []
    for _ in range(lanes):
        a, bb, w = struct.unpack_from('<IIf', body, off); off += 12
        L.append((a, bb, w))
    return {"n": n, "nodes": nodes, "lanes": lanes, "L": L, "consumed": off,
            "rest": body[off:]}

A = parse(sys.argv[1]); B = parse(sys.argv[2])
print(f"nodes: {A['n']} vs {B['n']} | lanes: {A['lanes']} vs {B['lanes']}")
print(f"consumed: {A['consumed']} vs {B['consumed']} | rest equal: {A['rest']==B['rest']} "
      f"(rest lens {len(A['rest'])} {len(B['rest'])})")
mdiff = [(A['nodes'][i][0], A['nodes'][i][1], B['nodes'][i][1])
         for i in range(min(A['n'], B['n'])) if A['nodes'][i][1] != B['nodes'][i][1]]
print("mass diffs:", len(mdiff), mdiff[:5])
ldiff = [(A['L'][i], A['L'][i][2], B['L'][i][2])
         for i in range(min(A['lanes'], B['lanes'])) if A['L'][i] != B['L'][i]]
print("lane diffs:", len(ldiff), ldiff[:5])
