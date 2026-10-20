"""Issue 014: for a padded fold at the reference target, score every single padded step of
every folding parameter from the final folding (one parameter moved to a padded value, the
rest kept) and print the deltas: cycles, LUT/BRAM/URAM/DSP, the polish's cost and the
annealer's DWC heuristic. Usage: python polish_probe.py <ref est dir> <board> [seed]"""
import copy
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.set_folding import SetFolding

ref_dir, board = sys.argv[1:3]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
perf = json.load(open(os.path.join(ref_dir, "report/estimate_network_performance.json")))
target = int(perf["max_cycles"])
start = ModelWrapper(os.path.join(ref_dir, "intermediate_models/phase_convert_to_hardware.onnx"))
probe_rows = []
orig_refine = SetFolding._refine


def refine_and_probe(self, opt, targets, target_cycles):
    out = orig_refine(self, opt, targets, target_cycles)
    if opt.padding == 0:
        return out
    params = opt.params.parameters
    base = opt.params.get_meta_values()
    local = self._local_scorer(opt)

    def evaluate(values):
        opt.params.set_values(values)
        opt.params.apply_updates(final=False, attrs=opt.PARAMETERS_TO_APPLY)
        if local is None:
            _, achieved, metrics = self._score(opt.model, opt)
        else:
            _, achieved, metrics = local()
        cost = self._cost(metrics, targets)
        fifo = opt.resource_cost(opt.fifo_resources(opt.model, opt.graph_nodes()))
        dwc = opt.compute_hls_dwc_cost(opt.model, opt.graph_nodes(), targets["LUT"])
        return achieved, metrics, cost, fifo, dwc

    b = evaluate(base)
    for i, p in enumerate(params):
        if p.name not in ("SIMD", "PE"):
            continue
        for v, real in zip(p.possible_values, p.real_values):
            padded = any(
                q.bound_name not in (None, "Kernel")
                and q.bound_value != q.node.get_nodeattr(q.bound_name)
                for q in real
            )
            bounds = [(q.bound_name, q.bound_value) for q in real if q.bound_name]
            cur = params[i].real_values[params[i].possible_values.index(base[i])]
            cur_bounds = [(q.bound_name, q.bound_value) for q in cur if q.bound_name]
            if bounds == cur_bounds:
                continue  # unpadded alternative (same channel counts as now)
            moved = list(base)
            moved[i] = v
            r = evaluate(moved)
            row = dict(
                node=p.unique_nodes[-1].onnx_node.name,
                param=p.name,
                frm=base[i],
                to=v,
                bounds=bounds,
                cycles=r[0],
                d_res={k: r[1].get(k, 0) - b[1].get(k, 0) for k in ("LUT", "BRAM_18K", "URAM", "DSP")},
                d_cost=r[2] - b[2],
                d_fifo=r[3] - b[3],
                d_dwc_heur=r[4] - b[4],
            )
            if r[0] <= target_cycles:
                probe_rows.append(row)
    evaluate(base)
    opt.params.apply_updates(final=True, attrs=opt.PARAMETERS_TO_APPLY)
    return out


SetFolding._refine = refine_and_probe
folder = SetFolding(target, platform=board, folding_maximum_padding=4, seed=seed, folding_effort=50)
folder.apply(copy.deepcopy(start))
print("target", target, "feasible single padded moves:", len(probe_rows))
probe_rows.sort(key=lambda r: r["d_cost"] + r["d_fifo"])
for r in probe_rows[:40]:
    print(
        f"{r['node']:32s} {r['param']:4s} {str(r['frm']):>14s}->{str(r['to']):14s} {r['bounds']} cyc {r['cycles']} "
        f"dres {r['d_res']} dcost {r['d_cost']:+.5f} dfifo {r['d_fifo']:+.5f} ddwc {r['d_dwc_heur']:+.5f}"
    )
