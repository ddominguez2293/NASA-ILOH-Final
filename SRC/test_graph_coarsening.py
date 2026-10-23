#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_graph_coarsening.py
#
# Purpose : regression test for build_coarsened_edges() (SRC/graph_utils.py),
#           verifying the breadth-first fix for GitHub issue #10A against
#           small synthetic networks -- a linear chain, a confluence
#           (two upstream branches merging), and a divergence (one upstream
#           node splitting into two downstream branches) -- per Matt's
#           explicit request to check exactly those three topologies.
# Usage   : python SRC/test_graph_coarsening.py

import pandas as pd

from graph_utils import build_coarsened_edges


def _edges(pairs):
    return pd.DataFrame(pairs, columns=["comid_up", "comid_dn"])


def _check(name, result_df, expected_edges):
    got = set(result_df[["comid_up", "comid_dn"]].itertuples(index=False, name=None))
    expected = set(expected_edges)
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got={sorted(got)} expected={sorted(expected)}")
    return ok


def test_linear_chain():
    # A -> B -> C -> D, only A and D are observed/modeled.
    # Nearest downstream observed node from A is D (B, C are unobserved pass-throughs).
    edges = _edges([(1, 2), (2, 3), (3, 4)])
    observed = {1, 4}
    result = build_coarsened_edges(edges, observed, max_hops=10, mode="nearest")
    return _check("linear chain", result, [(1, 4)])


def test_confluence():
    # A -> C, B -> C, C -> D. A, B, D observed; C is an unobserved confluence.
    # Both A and B should each independently find D as their nearest downstream.
    edges = _edges([(1, 3), (2, 3), (3, 4)])
    observed = {1, 2, 4}
    result = build_coarsened_edges(edges, observed, max_hops=10, mode="nearest")
    return _check("confluence", result, [(1, 4), (2, 4)])


def test_divergence():
    # A -> B, A -> C (A branches into two downstream paths), B -> D, C -> E.
    # A, D, E observed; B, C unobserved. Both branches must be found -- this
    # is exactly the case the old frontier-wipe-on-first-hit bug broke: it
    # would find only one of (A,D) / (A,E) and silently drop the other.
    edges = _edges([(1, 2), (1, 3), (2, 4), (3, 5)])
    observed = {1, 4, 5}
    result = build_coarsened_edges(edges, observed, max_hops=10, mode="nearest")
    return _check("divergence", result, [(1, 4), (1, 5)])


def test_divergence_uneven_depth():
    # A -> B (observed immediately), A -> C -> D (observed two hops later).
    # Both branches must still each report their own nearest node, even
    # though one is found much sooner than the other.
    edges = _edges([(1, 2), (1, 3), (3, 4)])
    observed = {1, 2, 4}
    result = build_coarsened_edges(edges, observed, max_hops=10, mode="nearest")
    return _check("divergence, uneven depth", result, [(1, 2), (1, 4)])


def test_nearest_picks_shortest_path():
    # A -> B -> C, and A -> D directly. C and D both observed; D is 1 hop
    # from A, C is 2 hops via B. "nearest" must pick D for that branch, and
    # B->C is a separate branch that correctly resolves to C.
    edges = _edges([(1, 2), (2, 3), (1, 4)])
    observed = {1, 3, 4}
    result = build_coarsened_edges(edges, observed, max_hops=10, mode="nearest")
    return _check("branch picks its own nearest node", result, [(1, 4), (1, 3)])


def test_no_path_to_any_observed_node():
    # A -> B -> C, none of B/C observed, A observed with no eligible downstream.
    # Should produce no edges for A rather than erroring.
    edges = _edges([(1, 2), (2, 3)])
    observed = {1}
    result = build_coarsened_edges(edges, observed, max_hops=10, mode="nearest")
    return _check("no eligible downstream node", result, [])


if __name__ == "__main__":
    results = [
        test_linear_chain(),
        test_confluence(),
        test_divergence(),
        test_divergence_uneven_depth(),
        test_nearest_picks_shortest_path(),
        test_no_path_to_any_observed_node(),
    ]
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{n_pass}/{n_total} graph coarsening tests passed.")
    if n_pass != n_total:
        raise SystemExit(1)
