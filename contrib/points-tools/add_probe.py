"""Add brute-force probe params to a trained .zlc (encoder-only; frames stay decodable by any zli).

usage: add_probe.py in.zlc out.zlc [--inner BYTES] [--inner-topk K] [--top BYTES] [--top-topk K]
--inner sets the probe on the ACE brute-force selectors (zl.ace#*), --top on the top-level brute force.
"""
import argparse, copy, cbor2

PROBE, TOPK = 7313, 7314

def add_probe(z, inner=None, inner_topk=2, top=None, top_topk=1):
    z = copy.deepcopy(z)
    if top:
        pid = f"probe{top}k{top_topk}"
        z["params"][pid] = {"ints": {PROBE: top, TOPK: top_topk}, "blobs": {}}
        z["graphs"][z["graphs"][z["start"]]["graphs"][0]]["params"] = pid
    if inner:
        for name, g in z["graphs"].items():
            if name.startswith("zl.ace#"):
                p = copy.deepcopy(z["params"][g["params"]])
                p["ints"][PROBE], p["ints"][TOPK] = inner, inner_topk
                pid = f"{g['params']}_p{inner}k{inner_topk}"
                z["params"][pid] = p
                g["params"] = pid
    return z

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("src"); a.add_argument("dst")
    a.add_argument("--inner", type=int); a.add_argument("--inner-topk", type=int, default=2)
    a.add_argument("--top", type=int); a.add_argument("--top-topk", type=int, default=1)
    o = a.parse_args()
    z = cbor2.loads(open(o.src, "rb").read())
    open(o.dst, "wb").write(cbor2.dumps(add_probe(z, o.inner, o.inner_topk, o.top, o.top_topk)))
