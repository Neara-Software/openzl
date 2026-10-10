"""Add brute-force narrowing params to a trained .zlc (encoder-only; frames stay decodable by any zli).

usage: add_narrow.py in.zlc out.zlc KEEP [KEEP ...]
Every ACE brute force selector (zl.ace#*) tries all successors on the first input of a file,
then keeps the KEEP[0] best, after the second input the KEEP[1] best, and so on.
The caller resets with ZL_CCtx_resetBruteForceNarrowing() between files.
"""
import copy, sys, cbor2

NARROW_BASE = 7300

def add_narrow(z, keeps):
    z = copy.deepcopy(z)
    for name, g in z["graphs"].items():
        if name.startswith("zl.ace#"):
            p = copy.deepcopy(z["params"][g["params"]])
            for s, k in enumerate(keeps):
                p["ints"][NARROW_BASE + s] = k
            pid = f"{g['params']}_n{'-'.join(map(str, keeps))}"
            z["params"][pid] = p
            g["params"] = pid
    return z

if __name__ == "__main__":
    src, dst, *keeps = sys.argv[1:]
    z = cbor2.loads(open(src, "rb").read())
    open(dst, "wb").write(cbor2.dumps(add_narrow(z, [int(k) for k in keeps])))
