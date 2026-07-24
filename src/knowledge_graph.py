"""Phase 3 (part 1) - Knowledge graph construction with NetworkX (docs/03).

Builds a directed graph of post-harvest domain triples (subject-predicate-object)
and persists it to ``data/kg/nodes.csv`` and ``data/kg/edges.csv`` for inspection
and for the Streamlit app.
"""
from __future__ import annotations

import os

import networkx as nx
import pandas as pd

from . import config as C


def build_graph() -> nx.DiGraph:
    """Construct the post-harvest knowledge graph from the rule specifications."""
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
