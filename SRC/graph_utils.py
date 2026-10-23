#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# graph_utils.py
#
# Purpose : shared graph-construction functions for the routed river network
#           (upstream->downstream COMID edges, optional coarsening to the
#           modeled/observed node set, and lightweight edge QA). Imported by
#           both DDNN_Model.py and GraphSHAP.py so the two scripts always
#           reconstruct the identical graph topology for a given domain --
#           this module used to be copy-pasted into both files, which is how
#           they drifted out of sync (GitHub issue #10A). There is nothing
#           CLI-specific or stateful here, so it's a plain importable module
#           rather than a top-level script.
#
# Comment style used in every script:
#   ## N. Section   = major section of the script
#   ### Step        = what the next block of code does
#   #### Note       = why / assumptions / caveats for that step
#   # inline        = short single-line bits next to the code

## 1. Imports
from collections import deque

import numpy as np
import pandas as pd


## 2. Edge construction

### Build upstream -> downstream COMID edge pairs from fromnode / tonode
def build_edge_pairs_from_nodes(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct upstream->downstream COMID edges by joining on node ids:
      (comid_up, tonode) matches (comid_dn, fromnode)
    Returns edge_pairs_all with columns [comid_up, comid_dn] (deduped).
    """
    edges_df = edges_df.dropna(subset=["fromnode", "tonode", "comid"]).copy()
    up = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_up"})
    dn = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_dn"})

    edge_pairs_all = (
        up.merge(dn, left_on="tonode", right_on="fromnode", how="inner")[["comid_up", "comid_dn"]]
          .drop_duplicates()
          .reset_index(drop=True)
    )
    return edge_pairs_all


### Coarsen the network to edges between observed COMIDs
def build_coarsened_edges(edge_pairs_all: pd.DataFrame, observed_comids: set,
                          max_hops=5000, mode="nearest") -> pd.DataFrame:
    """
    Collapse the full COMID network down to edges between observed COMIDs only.

    For each observed COMID, walk downstream through the full graph until you hit another
    observed COMID. Add an edge (src_observed -> downstream_observed).
      mode="nearest": each downstream branch stops at its own first-encountered
      observed node (true breadth-first "nearest"), rather than the first hit
      anywhere terminating every other branch too.

    Fixes GitHub issue #10A. The previous version used frontier.pop() (a
    stack -- depth-first, not breadth-first) and, in "nearest" mode, cleared
    the *entire* frontier and broke out on the first observed node found by
    *any* branch. That silently dropped legitimate downstream connections
    for every other in-flight branch (e.g. a sibling path from an earlier
    divergence that hadn't reached its own nearest observed node yet), and
    wasn't guaranteed to find the nearest node along the branch that did hit,
    since depth-first exploration doesn't visit nodes in distance order.

    Each element of the search frontier below is one independent branch:
    (current_node, hops_so_far, nodes_visited_on_this_branch). A branch stops
    expanding as soon as it reaches any observed node (its nearest, since
    breadth-first visits in increasing-hop order) without touching any other
    branch's search. Branches are explored breadth-first via a FIFO deque.

    Verified against synthetic linear / confluence / divergence networks in
    test_graph_coarsening.py.
    """
    dn_map = {}
    for u, v in edge_pairs_all[["comid_up", "comid_dn"]].itertuples(index=False):
        try:
            u_i = int(u); v_i = int(v)
        except Exception:
            continue
        dn_map.setdefault(u_i, []).append(v_i)

    coarse_edges = set()

    for src in observed_comids:
        src = int(src)

        # Seed one independent branch per immediate downstream neighbor of src,
        # so a divergence right at src is tracked as separate branches too.
        frontier = deque(
            (dn, 1, frozenset({src, dn})) for dn in dn_map.get(src, [])
        )
        found_targets = set()

        while frontier:
            curr, hops, visited = frontier.popleft()
            if hops > max_hops:
                continue

            if curr in observed_comids and curr != src:
                found_targets.add(curr)
                if mode == "nearest":
                    continue  # this branch found its nearest node; stop expanding it
                # mode != "nearest" falls through to keep walking past observed nodes too

            for dn in dn_map.get(curr, []):
                if dn in visited:
                    continue
                frontier.append((dn, hops + 1, visited | {dn}))

        for tgt in found_targets:
            coarse_edges.add((src, tgt))

    return pd.DataFrame(list(coarse_edges), columns=["comid_up", "comid_dn"])


## 3. Edge QA checks
def edge_qa_checks(edge_pairs: pd.DataFrame, all_comids: np.ndarray) -> None:
    """Lightweight QA checks for edge health."""
    print("\n====================")
    print("EDGE QA CHECKS")
    print("====================")

    n_nodes = len(all_comids)
    n_edges = len(edge_pairs)

    print(f"Nodes in modeling set: {n_nodes}")
    print(f"Edges after filtering/coarsening: {n_edges}")
    print(f"Edge density: {n_edges / max(n_nodes,1):.4f} edges/node")

    if n_edges == 0:
        print("WARNING: No edges in graph. Model will fail because it requires edge_index.")
        return

    self_loops = int((edge_pairs["comid_up"].astype(int) == edge_pairs["comid_dn"].astype(int)).sum())
    dup_edges = int(edge_pairs.duplicated(subset=["comid_up", "comid_dn"]).sum())

    nodes_up = set(edge_pairs["comid_up"].astype(int).unique())
    nodes_dn = set(edge_pairs["comid_dn"].astype(int).unique())
    nodes_any = nodes_up | nodes_dn

    print(f"Nodes as upstream:   {len(nodes_up)} ({100*len(nodes_up)/n_nodes:.2f}%)")
    print(f"Nodes as downstream: {len(nodes_dn)} ({100*len(nodes_dn)/n_nodes:.2f}%)")
    print(f"Nodes in any edge:   {len(nodes_any)} ({100*len(nodes_any)/n_nodes:.2f}%)")
    print(f"Self-loops: {self_loops}")
    print(f"Duplicate edges: {dup_edges}")
    print("EDGE QA DONE.")
