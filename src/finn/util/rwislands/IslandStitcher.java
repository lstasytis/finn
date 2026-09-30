/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

import com.xilinx.rapidwright.design.Design;
import com.xilinx.rapidwright.design.DesignTools;
import com.xilinx.rapidwright.design.Net;
import com.xilinx.rapidwright.design.SitePinInst;
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
 * usage: IslandStitcher top.dcp top.edf out.dcp maxIter threads cell=island.dcp,island.edf ...
 *
 * top.dcp holds the accelerator netlist with one black box per island. Each black box is
 * filled with its island's implementation (the equivalent of Vivado's read_checkpoint -cell),
 * the nets between islands are routed with RWRoute (partial routing: the islands' own routing
 * stays as it is). Nets to or from the accelerator's top-level ports and the clock are left
 * for Vivado, which routes them when the accelerator is inserted into the shell.
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
        for (int i = 5; i < args.length; i++) {
            String[] kv = args[i].split("=", 2);
            islands.put(kv[0], kv[1].split(","));
        }

        // reading the island checkpoints dominates for many islands: in parallel
        ExecutorService ex = Executors.newFixedThreadPool(Math.max(1, threads));
        Future<Design> topF = ex.submit(() -> Design.readCheckpoint(args[0], args[1]));
        Map<String, Future<Design>> islF = new LinkedHashMap<>();
        for (Map.Entry<String, String[]> e : islands.entrySet()) {
            String[] f = e.getValue();
            islF.put(e.getKey(), ex.submit(() -> Design.readCheckpoint(f[0], f[1])));
        }
        Design top = topF.get();
        stamp("read_top");
        for (Map.Entry<String, Future<Design>> e : islF.entrySet()) {
            DesignTools.populateBlackBox(top, e.getKey(), e.getValue().get());
        }
        ex.shutdown();
        stamp("populate");

        DesignTools.makePhysNetNamesConsistent(top);
        DesignTools.createMissingSitePinInsts(top);
        // pins between islands: nets with a driver inside the accelerator and unrouted sinks
        List<SitePinInst> pins = new ArrayList<>();
        int nets = 0;
        for (Net net : top.getNets()) {
            if (net.isClockNet() || net.isStaticNet() || net.getSource() == null) continue;
            boolean any = false;
            for (SitePinInst p : net.getSinkPins()) {
                if (!p.isRouted()) {
                    pins.add(p);
                    any = true;
                }
            }
            if (any) nets++;
        }
        System.out.println("INFO: routing " + pins.size() + " pins of " + nets + " nets between islands");
        stamp("prepare_route");

        List<String> a = new ArrayList<>();
        a.add("--fixBoundingBox");
        a.add("--useUTurnNodes");
        a.add("--nonTimingDriven");
        a.add("--maxIterations");
        a.add(Integer.toString(maxIter));
        top = PartialRouter.routeDesignWithUserDefinedArguments(top, a.toArray(new String[0]), pins, false);
        stamp("route");

        int unrouted = 0;
        for (SitePinInst p : pins) {
            if (!p.isRouted()) unrouted++;
        }
        System.out.println("RESULT unrouted_pins " + unrouted);
        top.writeCheckpoint(out);
        stamp("write");
    }
}
