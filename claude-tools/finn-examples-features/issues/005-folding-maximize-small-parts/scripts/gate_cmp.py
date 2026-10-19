import json, os, sys, glob
a_dir, b_dir = sys.argv[1:3]
same = diff = 0
for fa in sorted(glob.glob(a_dir + "/*.json")):
    k = os.path.basename(fa)
    if k.startswith("_"):
        continue
    fb = os.path.join(b_dir, k)
    if not os.path.exists(fb):
        print("MISSING", k); continue
    A, B = json.load(open(fa)), json.load(open(fb))
    if A == B:
        same += 1; continue
    diff += 1
    def s(D):
        if D.get("result") != "ok":
            return D.get("result"), D.get("error", "")[:80]
        r = D.get("search_report") or {}
        rb = r.get("resources_as_built", {})
        return (D["performance"]["max_cycles"], r.get("fits_budget"),
                {x: round(v) for x, v in rb.items()})
    na = {n["name"]: n["attrs"] for n in A.get("graph", {}).get("nodes", [])}
    nb = {n["name"]: n["attrs"] for n in B.get("graph", {}).get("nodes", [])}
    keys = sorted({(kk) for n in na if n in nb for kk in set(na[n]) | set(nb[n]) if na[n].get(kk) != nb[n].get(kk)})
    print(f"{k[:-5]:55s} base {s(A)}\n{'':55s} fix  {s(B)}  attrs: {keys}")
print("same", same, "diff", diff)
