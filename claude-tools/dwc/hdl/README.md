# Recovered generalized-DWC RTL

These were untracked working files at repo root / `finn-rtllib/dwc/hdl/` that were
lost during branch restructuring (2026-07-24) and recovered here for future use.

| file | lines | source |
|---|---|---|
| `dwc_generalized.sv` | 141 | Claude Code file-history (session `fbecec92`) — the 300 MHz / II=1 core; newer than `../dwc_generalized.vpc_backup.sv` (132 lines) |
| `dwc_generalized_axi.sv` | 40 | Claude Code file-history (session `fbecec92`) — AXI-Stream adapter around the core |
| `vpc.sv` | 152 | copy of `../ref_vpc/vpc.sv` — Preußer Vector Pack Converter reference |

To use in a build, copy `dwc_generalized*.sv` into `finn-rtllib/dwc/hdl/`. They are
kept here (not in `finn-rtllib/`) so they travel with the tools and are never
untracked again. See memory `dwc-generalized-rtl-verified` / `dwc-300mhz-rtl-track`.
