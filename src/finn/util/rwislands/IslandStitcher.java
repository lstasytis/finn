/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

import com.xilinx.rapidwright.design.Cell;
import com.xilinx.rapidwright.design.Design;
import com.xilinx.rapidwright.design.DesignTools;
import com.xilinx.rapidwright.design.Net;
import com.xilinx.rapidwright.design.SitePinInst;
import com.xilinx.rapidwright.edif.EDIFCellInst;
import com.xilinx.rapidwright.edif.EDIFHierCellInst;
import com.xilinx.rapidwright.edif.EDIFHierNet;
import com.xilinx.rapidwright.edif.EDIFHierPortInst;
import com.xilinx.rapidwright.edif.EDIFPortInst;
import com.xilinx.rapidwright.edif.EDIFNet;
import com.xilinx.rapidwright.edif.EDIFNetlist;
import com.xilinx.rapidwright.rwroute.PartialRouter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Stitches FINN islands that were placed and routed out of context at their final location
 * (no relocation) into the accelerator netlist.
 *
 * usage: IslandStitcher top.dcp top.edf out.dcp maxIter threads [--region=ranges]
 *        [--shell=shell.dcp,shell.edf --shellcell=hier/name --boundary=file] cell=island.dcp,island.edf ...
 *
 * --region: site ranges (e.g. "SLICE_X60Y0:SLICE_X120Y359 ...") the inter-island routes must stay
 * in (RWRoute --pblock): the stitched design does not contain the shell, whose locked routing
 * would otherwise be overlapped where a stitch route crosses the shell region.
 *
 * top.dcp holds the accelerator netlist with one black box per island. Each black box is
 * filled with its island's implementation (the equivalent of Vivado's read_checkpoint -cell),
 * the nets between islands are routed with RWRoute (partial routing: the islands' own routing
 * stays as it is). Nets to or from the accelerator's top-level ports and the clock are left
 * for Vivado, which routes them when the accelerator is inserted into the shell.
 *
 * --shell (final route in RapidWright): the implemented shell (Vivado checkpoint with the
 * accelerator as a black box at --shellcell, its EDIF exported by write_edif; encrypted IP comes
 * as .edn files next to it) is read in parallel with the islands; the accelerator netlist and
 * the islands are inserted into the shell's black box, and one RWRoute pass routes everything
 * still open: the nets between islands, the shell/accelerator boundary, the accelerator's clock
 * loads (incremental clock routing from the shell's clock tree) and new static pins. out.dcp is
 * then the complete design: Vivado only loads it and writes the bitstream (no route_design,
 * whose fixed cost on a 1M-net design is ~10 min even with nothing left to route).
 */
public class IslandStitcher {
    private static long t0 = System.nanoTime();

    private static void stamp(String what) {
        System.out.printf("STAMP %s %.3f%n", what, (System.nanoTime() - t0) / 1e9);
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 6) {
            System.out.println("usage: IslandStitcher top.dcp top.edf out.dcp maxIter threads cell=dcp,edf ...");
            System.exit(1);
        }
        String out = args[2];
        int maxIter = Integer.parseInt(args[3]);
        int threads = Integer.parseInt(args[4]);
        Map<String, String[]> islands = new LinkedHashMap<>();
        String region = null;
        String[] shellFiles = null;
        String shellCell = null;
        String boundary = null;
        for (int i = 5; i < args.length; i++) {
            if (args[i].startsWith("--region=")) {
                region = args[i].substring("--region=".length());
                continue;
            }
            if (args[i].startsWith("--shell=")) {
                shellFiles = args[i].substring("--shell=".length()).split(",");
                continue;
            }
            if (args[i].startsWith("--boundary=")) {
                boundary = args[i].substring("--boundary=".length());
                continue;
            }
            if (args[i].startsWith("--shellcell=")) {
                shellCell = args[i].substring("--shellcell=".length());
                continue;
            }
            String[] kv = args[i].split("=", 2);
            islands.put(kv[0], kv[1].split(","));
        }

        // reading the island checkpoints dominates for many islands: in parallel
        ExecutorService ex = Executors.newFixedThreadPool(Math.max(1, threads));
        Future<Design> topF = ex.submit(() -> Design.readCheckpoint(args[0], args[1]));
        final String[] sf = shellFiles;
        Future<Design> shellF = sf == null ? null : ex.submit(() -> Design.readCheckpoint(sf[0], sf[1]));
        Map<String, Future<Design>> islF = new LinkedHashMap<>();
        for (Map.Entry<String, String[]> e : islands.entrySet()) {
            String[] f = e.getValue();
            islF.put(e.getKey(), ex.submit(() -> Design.readCheckpoint(f[0], f[1])));
        }
        Design top = topF.get();
        stamp("read_top");
        // the design the islands go into: the accelerator alone, or (final route) the shell with
        // the accelerator netlist in its black box
        Design target = top;
        String prefix = "";
        if (shellF != null) {
            target = shellF.get();
            stamp("read_shell");
            fill(target, shellCell, top, "accelerator");
            prefix = shellCell + "/";
            stamp("populate_shell");
        }
        for (Map.Entry<String, Future<Design>> e : islF.entrySet()) {
            fill(target, prefix + e.getKey(), e.getValue().get(), "island");
        }
        ex.shutdown();
        stamp("populate");

        DesignTools.makePhysNetNamesConsistent(target);
        stamp("names");
        java.util.Set<Net> joinedNets = new java.util.HashSet<>();
        if (shellF != null && boundary != null) {
            joinedNets = joinBoundary(target, shellCell, boundary);
            stamp("boundary");
        }
        if (shellF != null) {
            reunite(target);
            stamp("reunite");
        }
        // the islands' own nets are complete; only the nets of the accelerator's top cell
        // (between islands, to the top-level ports) need site pins and routing
        EDIFNetlist nl = target.getNetlist();
        EDIFHierCellInst accInst = shellF == null ? nl.getTopHierCellInst() : nl.getHierCellInstFromName(shellCell);
        List<SitePinInst> pins = new ArrayList<>();
        int nets = 0;
        // 1. candidate nets: the accelerator's top-cell nets (by parent-net name), and in the final
        // route the nets at the black box's ports and the joined boundary nets
        java.util.LinkedHashSet<Net> cand = new java.util.LinkedHashSet<>();
        for (EDIFNet en : accInst.getCellType().getNets()) {
            EDIFHierNet parent = nl.getParentNet(new EDIFHierNet(accInst, en));
            if (parent == null) continue;
            Net net = target.getNet(parent.getHierarchicalNetName());
            if (net == null || net.isStaticNet()) continue;
            // the clock: in the final route its new loads (incremental clock routing)
            if (net.isClockNet() && shellF == null) continue;
            cand.add(net);
        }
        int noNet = 0;
        if (shellF != null) {
            EDIFHierCellInst parentInst = accInst.getParent();
            for (EDIFPortInst pi : accInst.getInst().getPortInsts()) {
                Net net = nl.getPhysicalNetFromPin(new EDIFHierPortInst(parentInst, pi), target);
                if (net == null) {
                    noNet++;
                } else if (!net.isStaticNet()) {
                    cand.add(net);
                }
            }
            cand.addAll(joinedNets);
        }
        // 2. site pins; in the final route one physical net per logical net (unify)
        int unified = 0;
        for (Net net : cand) {
            if (target.getNet(net.getName()) != net) continue;
            DesignTools.createMissingSitePinInsts(target, net);
            if (shellF != null) unified += unify(target, net);
        }
        // 3. the unrouted sinks of the nets left
        int noSource = 0;
        for (Net net : cand) {
            if (target.getNet(net.getName()) != net) continue;
            if (net.getSource() == null) {
                noSource++;
                continue;
            }
            if (shellF != null) DesignTools.updatePinsIsRouted(net);
            boolean any = false;
            for (SitePinInst p : net.getSinkPins()) {
                if (!p.isRouted()) {
                    pins.add(p);
                    any = true;
                }
            }
            if (any) nets++;
        }
        System.out.println("INFO: candidate nets " + cand.size() + ", merged " + unified + ", without source " + noSource
                + ", black-box ports without physical net " + noNet);
        if (shellF != null) {
            // static pins without routing: the merge drops part of the shell's static routing
            for (Net sn : new Net[] {target.getGndNet(), target.getVccNet()}) {
                DesignTools.updatePinsIsRouted(sn);
                int k = 0;
                for (SitePinInst p : sn.getPins()) {
                    if (!p.isOutPin() && !p.isRouted()) {
                        pins.add(p);
                        k++;
                    }
                }
                System.out.println("INFO: " + k + " unrouted pins of " + sn.getName());
            }
        }
        stamp("site_pins");
        {
            int stale = 0;
            java.util.Set<Net> staleNets = new java.util.HashSet<>();
            for (SitePinInst p : pins) {
                Net n = p.getNet();
                if (target.getNet(n.getName()) != n) {
                    stale++;
                    if (staleNets.add(n) && staleNets.size() <= 3) {
                        Net r = target.getNet(n.getName());
                        System.out.println("STALE " + n + " registered " + r + (r == null ? "" : " pins " + r.getPins() + " pips " + r.getPIPs().size()));
                    }
                }
            }
            System.out.println("INFO: pins on unregistered nets before routing " + stale + " (" + staleNets.size() + " nets)");
        }
        System.out.println("INFO: routing " + pins.size() + " pins of " + nets + " nets"
                + (shellF == null ? " between islands" : " (islands, shell boundary, clock, static)"));
        stamp("prepare_route");

        List<String> a = new ArrayList<>();
        a.add("--fixBoundingBox");
        a.add("--useUTurnNodes");
        a.add("--nonTimingDriven");
        a.add("--maxIterations");
        a.add(Integer.toString(maxIter));
        // (the final route knows the shell's routing: no region)
        if (shellF == null && region != null && !region.isEmpty()) {
            a.add("--pblock");
            a.add(region);
        }
        // soft preserve: an island's own routing may box in one of its boundary pins (all
        // access nodes used); RWRoute then rips up and re-routes the blocking island nets instead
        // of giving up on the pin (VGG10, 20 islands: 56 pins left unrouted, which the assembly's
        // interactive router could not finish either)
        target = PartialRouter.routeDesignWithUserDefinedArguments(target, a.toArray(new String[0]), pins, true);
        stamp("route");

        // pins left over: a slice input boxed in by its island's routing, which the fixed
        // bounding box and one unpreserve round cannot reach (VGG10 8x: 3 pins). Again on their
        // own, with a growing bounding box and more iterations
        List<SitePinInst> left = new ArrayList<>();
        for (SitePinInst p : pins) {
            if (!p.isRouted()) left.add(p);
        }
        if (!left.isEmpty()) {
            System.out.println("INFO: second pass for " + left.size() + " pins");
            List<String> b = new ArrayList<>();
            b.add("--useUTurnNodes");
            b.add("--nonTimingDriven");
            b.add("--maxIterations");
            b.add(Integer.toString(Math.max(30, 3 * maxIter)));
            if (shellF == null && region != null && !region.isEmpty()) {
                b.add("--pblock");
                b.add(region);
            }
            target = PartialRouter.routeDesignWithUserDefinedArguments(target, b.toArray(new String[0]), left, true);
            stamp("route_retry");
        }

        int unrouted = 0;
        java.util.Set<String> unroutedNets = new java.util.TreeSet<>();
        for (SitePinInst p : pins) {
            if (!p.isRouted()) {
                if (unrouted < 2) {
                    Net n = p.getNet();
                    com.xilinx.rapidwright.device.Node sn = n.getSource() == null ? null : n.getSource().getConnectedNode();
                    for (Net o : shellF == null ? new ArrayList<Net>() : target.getNets()) {
                        if (o == n || sn == null) continue;
                        for (com.xilinx.rapidwright.device.PIP pp : o.getPIPs()) {
                            if (pp.getStartNode().equals(sn) || pp.getEndNode().equals(sn)) {
                                System.out.println("SOURCE_NODE_USED_BY " + o + " pins " + o.getPins() + " registered " + (target.getNet(o.getName()) == o));
                                break;
                            }
                        }
                    }
                    System.out.println("UNROUTED_PIN " + p + " net " + n + " registered " + (target.getNet(n.getName()) == n) + " type " + n.getType() + " source " + n.getSource()
                            + " altsource " + n.getAlternateSource() + " pips " + n.getPIPs().size()
                            + " pins " + n.getPins() + " hierport " + DesignTools.isNetDrivenByHierPort(n));
                }
                unrouted++;
                unroutedNets.add(p.getNet().getName());
            }
        }
        System.out.println("RESULT unrouted_pins " + unrouted);
        for (String n : unroutedNets) {
            System.out.println("UNROUTED_NET " + n);
        }
        // the islands' netlists come with their own libraries (xil_defaultlib, work_<node>):
        // one work library, so that no library refers to one written after it
        target.getNetlist().consolidateAllToWorkLibrary(true);
        // the islands' static (VCC/GND) routing must stay changeable: Vivado has to rip up
        // parts of it where boundary nets need the same site pins / nodes at assembly
        target.getGndNet().unlockRouting();
        target.getVccNet().unlockRouting();
        target.writeCheckpoint(out);
        stamp("write");
    }

    /**
     * Joins the shell/accelerator boundary nets. Most of them are driven or loaded inside
     * encrypted shell IP (SmartConnect, AXI interconnect), through which RapidWright cannot follow
     * the netlist: the shell side is a physical net without site pins, named by Vivado after its
     * real driver, the accelerator side a net named after the black-box port. boundaryFile (from
     * Vivado, see flow.shell_for_rapidwright): "port IN|OUT site/pin" per shell leaf pin. The
     * accelerator side is found through the islands' (plain) leaf cells, merged into the shell's
     * net (whose name Vivado knows), and the shell's site pins are created.
     */
    private static java.util.Set<Net> joinBoundary(Design d, String cellName, String boundaryFile) throws Exception {
        EDIFNetlist nl = d.getNetlist();
        EDIFHierCellInst acc = nl.getHierCellInstFromName(cellName);
        String prefix = cellName + "/";
        // port name (bit) -> net inside the accelerator cell
        Map<String, EDIFNet> inner = new java.util.HashMap<>();
        for (EDIFNet en : acc.getCellType().getNets()) {
            for (EDIFPortInst pi : en.getPortInsts()) {
                if (pi.isTopLevelPort()) inner.put(pi.getName(), en);
            }
        }
        Map<String, List<String[]>> sitePins = new LinkedHashMap<>();
        for (String line : java.nio.file.Files.readAllLines(java.nio.file.Paths.get(boundaryFile))) {
            String[] f = line.trim().split("\\s+");
            if (f.length == 3) sitePins.computeIfAbsent(f[0], k -> new ArrayList<>()).add(f);
        }
        int joined = 0, noInner = 0, noAccNet = 0, noShellNet = 0;
        java.util.Set<Net> out = new java.util.HashSet<>();
        for (Map.Entry<String, List<String[]>> e : sitePins.entrySet()) {
            EDIFNet en = inner.get(e.getKey());
            if (en == null) {
                noInner++;
                continue;
            }
            // the accelerator's physical net: through a placed leaf cell inside the accelerator
            Net accNet = null;
            List<EDIFHierPortInst> leaves = nl.getPhysicalPins(nl.getParentNet(new EDIFHierNet(acc, en)));
            for (EDIFHierPortInst l : leaves == null ? new ArrayList<EDIFHierPortInst>() : leaves) {
                if (!l.getFullHierarchicalInstName().startsWith(prefix)) continue;
                Cell c = d.getCell(l.getFullHierarchicalInstName());
                if (c == null || c.getSiteInst() == null) continue;
                String sw = c.getSiteWireNameFromLogicalPin(l.getPortInst().getName());
                Net n = sw == null ? null : c.getSiteInst().getNetFromSiteWire(sw);
                if (n != null && !n.isStaticNet() && !n.isClockNet()) {
                    accNet = n;
                    break;
                }
            }
            if (accNet == null) {
                // (the net named after the port, as inserted by populateBlackBox)
                accNet = d.getNet(prefix + e.getKey());
            }
            if (accNet == null) {
                if (noAccNet++ < 5) {
                    System.out.println("INFO: no accelerator net for " + e.getKey() + " leaves " + leaves);
                }
                continue;
            }
            for (String[] f : e.getValue()) {
                String[] sp = f[2].split("/");
                com.xilinx.rapidwright.design.SiteInst si = d.getSiteInstFromSiteName(sp[0]);
                if (si == null) {
                    noShellNet++;
                    continue;
                }
                Net shellNet = si.getNetFromSiteWire(sp[1]);
                if (shellNet == null || shellNet.isStaticNet() || shellNet.isClockNet()) {
                    noShellNet++;
                    continue;
                }
                if (shellNet != accNet) {
                    d.movePinsToNewNetDeleteOldNet(accNet, shellNet, true);
                    accNet = shellNet;
                }
                if (si.getSitePinInst(sp[1]) == null) {
                    accNet.createPin(sp[1], si);
                }
            }
            joined++;
            out.add(accNet);
        }
        System.out.println("INFO: boundary nets joined " + joined + ", port not found " + noInner
                + ", no accelerator net " + noAccNet + ", shell site pins without net " + noShellNet);
        return out;
    }

    /**
     * Routed nets without pins: an island's port net (routed from its driver to the partition
     * pin at the island's edge) whose pins went to the net named after the accelerator's port
     * when the black boxes were filled, while its routing stayed on the island's net. RWRoute
     * then finds the source node used by another net and skips the connection silently. Each
     * such orphan is merged into the net that has the pin at the start of its routing.
     */
    private static void reunite(Design d) {
        Map<com.xilinx.rapidwright.device.Node, Net> start = new java.util.HashMap<>();
        for (Net n : d.getNets()) {
            if (n.isStaticNet() || n.isClockNet() || n.getPIPs().isEmpty() || n.getSource() != null) continue;
            for (com.xilinx.rapidwright.device.PIP p : n.getPIPs()) {
                start.put(p.getStartNode(), n);
            }
        }
        int merged = 0;
        if (!start.isEmpty()) {
            for (Net n : new ArrayList<>(d.getNets())) {
                SitePinInst src = n.getSource();
                if (src == null || n.isStaticNet() || n.isClockNet()) continue;
                Net orphan = start.get(src.getConnectedNode());
                if (orphan == null || orphan == n || d.getNet(orphan.getName()) != orphan) continue;
                d.movePinsToNewNetDeleteOldNet(orphan, n, true);
                merged++;
            }
        }
        System.out.println("INFO: routed nets without pins " + new java.util.HashSet<>(start.values()).size()
                + ", merged into the net of their driver " + merged);
    }

    /**
     * The site wires of a net's pins may still belong to other physical nets after the black
     * boxes were filled: the island's port net at the driver, the shell's net at a load (one
     * logical net, several physical names). RWRoute skips such a net silently; those nets are
     * merged into this one. Returns the number of merged nets.
     */
    private static int unify(Design d, Net net) {
        int k = 0;
        for (SitePinInst p : new ArrayList<>(net.getPins())) {
            Net w = p.getSiteInst().getNetFromSiteWire(p.getSiteWireName());
            if (w != null && w != net && !w.isStaticNet() && !w.isClockNet() && d.getNet(w.getName()) == w) {
                // (pins this net has already, e.g. created by createMissingSitePinInsts on both)
                java.util.Set<String> have = new java.util.HashSet<>();
                for (SitePinInst q : net.getPins()) have.add(q.getSiteInst().getName() + "/" + q.getName());
                for (SitePinInst q : new ArrayList<>(w.getPins())) {
                    if (have.contains(q.getSiteInst().getName() + "/" + q.getName())) w.removePin(q);
                }
                d.movePinsToNewNetDeleteOldNet(w, net, true);
                k++;
            }
        }
        return k;
    }

    /** read_checkpoint -cell: fill the black box at name with d. */
    private static void fill(Design target, String name, Design d, String what) {
        // Vivado marks black boxes with black_box = "true", which RapidWright's
        // EDIFCellInst.isBlackBox() does not recognize (it expects IS_IMPORTED or "1")
        EDIFCellInst inst = target.getNetlist().getCellInstFromHierName(name);
        if (inst == null) {
            throw new RuntimeException(what + " " + name + " not found");
        }
        inst.addProperty(EDIFCellInst.BLACK_BOX_PROP, "true");
        DesignTools.populateBlackBox(target, name, d);
        inst.removeProperty(EDIFCellInst.BLACK_BOX_PROP);
        if (inst.getCellType().isLeafCellOrBlackBox()) {
            throw new RuntimeException(what + " " + name + " was not filled");
        }
    }
}
