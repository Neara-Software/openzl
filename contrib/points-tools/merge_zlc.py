"""Merge the top-level brute-force candidates of two trained .zlc files into one compressor.

usage: merge_zlc.py a.zlc b.zlc out.zlc [--pick-a 0,1] [--pick-b 0,1]
Both inputs must start with segment_serial -> brute_force (the points-i32 profiles do).
Graph/node names of b are renumbered (+1000) so they do not clash with a.
"""
import argparse, cbor2

def rn(o, off):
    if isinstance(o, dict):
        return {rn(k, off) if isinstance(k, str) else k: rn(v, off) for k, v in o.items()}
    if isinstance(o, list):
        return [rn(v, off) for v in o]
    if isinstance(o, str) and "#" in o:
        a, b = o.rsplit("#", 1)
        if b.isdigit():
            return f"{a}#{int(b) + off}"
    return o

def merge(za, zb, pick_a, pick_b, off=1000):
    zb = {k: (rn(v, off) if k in ("nodes", "graphs", "start") else v) for k, v in zb.items()}
    m = {"version": za["version"], "global_params": za["global_params"], "mparams": {},
         "params": {**za["params"], **zb["params"]},
         "nodes": {**za["nodes"], **zb["nodes"]},
         "graphs": {**za["graphs"], **zb["graphs"]}}
    ta = za["graphs"][za["start"]]["graphs"][0]
    tb = zb["graphs"][zb["start"]]["graphs"][0]
    ga, gb = m["graphs"][ta]["graphs"], m["graphs"][tb]["graphs"]
    m["graphs"]["zl.private.brute_force#9000"] = {
        "type": "parameterized", "base": "zl.private.brute_force",
        "graphs": [ga[i] for i in pick_a] + [gb[i] for i in pick_b],
        "nodes": [], "params": za["graphs"][ta]["params"]}
    start = dict(za["graphs"][za["start"]])
    start["graphs"] = ["zl.private.brute_force#9000"]
    m["graphs"]["zl.segment_serial#9001"] = start
    m["start"] = "zl.segment_serial#9001"
    return m

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b"); p.add_argument("out")
    p.add_argument("--pick-a", default="0,1"); p.add_argument("--pick-b", default="0,1")
    o = p.parse_args()
    za = cbor2.loads(open(o.a, "rb").read()); zb = cbor2.loads(open(o.b, "rb").read())
    pa = [int(x) for x in o.pick_a.split(",") if x]; pb = [int(x) for x in o.pick_b.split(",") if x]
    open(o.out, "wb").write(cbor2.dumps(merge(za, zb, pa, pb)))
