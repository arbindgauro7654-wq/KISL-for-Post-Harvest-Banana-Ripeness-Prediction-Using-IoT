"""Phase 3 (part 1) - Knowledge graph construction with NetworkX (docs/03).

Builds a directed graph of post-harvest domain triples (subject-predicate-object)
and persists it to ``data/kg/nodes.csv`` and ``data/kg/edges.csv`` for inspection
and for the Streamlit app.

Viva tip: NetworkX here is for documentation/visualisation — the ML model uses
the tabular flags from kg_features.py, not graph traversal at inference time.
"""
from __future__ import annotations

import os

import networkx as nx   # Used for: directed graph of literature triples (DiGraph)
import pandas as pd     # Used for: export nodes/edges CSV for Knowledge Graph page

from . import config as C


def build_graph() -> nx.DiGraph:
    """Used for: turn KG_RULES + KG_INTERACTION_RULES into a visualisable triple graph."""
    G = nx.DiGraph()

    def _add(subject, predicate, obj, **attrs):
        G.add_node(subject, kind="condition")
        G.add_node(obj, kind="concept")
        G.add_edge(subject, obj, predicate=predicate, **attrs)

    for r in C.KG_RULES:
        cond = f"{r['subject']} {r['op']} threshold"
        _add(
            cond,
            r["predicate"],
            r["object"],
            rule_id=r["id"],
            source=r["source"],
            expected_dir=r["expected_dir"],
        )

    for r in C.KG_INTERACTION_RULES:
        cond = r["label"]
        _add(
            cond,
            r["predicate"],
            r["object"],
            rule_id=r["id"],
            source=r["source"],
            expected_dir=r["expected_dir"],
        )

    return G


def save_graph(G: nx.DiGraph) -> tuple[str, str]:
    """Used for: persist nodes.csv / edges.csv under data/kg/ for the app."""
    nodes = pd.DataFrame(
        [{"node": n, **d} for n, d in G.nodes(data=True)]
    )
    edges = pd.DataFrame(
        [
            {"subject": u, "object": v, **d}
            for u, v, d in G.edges(data=True)
        ]
    )
    nodes_path = os.path.join(C.KG_DIR, "nodes.csv")
    edges_path = os.path.join(C.KG_DIR, "edges.csv")
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    return nodes_path, edges_path


def summary(G: nx.DiGraph) -> str:
    lines = [
        "Knowledge Graph summary",
        "=======================",
        f"Nodes: {G.number_of_nodes()}",
        f"Edges (triples): {G.number_of_edges()}",
        "",
        "Triples:",
    ]
    for u, v, d in G.edges(data=True):
        lines.append(f"  [{d.get('rule_id','?')}] ({u}) -[{d['predicate']}]-> ({v})  {d.get('source','')}")
    return "\n".join(lines)
