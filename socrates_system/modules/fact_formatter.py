"""
GraphRAG-Inspired Fact Linearization
Converts session KG (nodes/edges) and high-confidence claims into a concise, readable summary
suited for contradiction detection prompts.
"""
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class GraphRAGFactFormatter:
    def format_session_facts(self, kg_export: Dict[str, Any], high_conf_claims: List[Dict[str, Any]], max_items: int = 50) -> str:
        """Return a GraphRAG-style linearized summary string of the session knowledge.
        Args:
            kg_export: output of KnowledgeGraphManager.export_session_graph(session_id)
            high_conf_claims: list of {text, confidence, evidence}
            max_items: limit per section to keep prompt concise
        """
        nodes = kg_export.get("nodes", {}) or {}
        edges = kg_export.get("edges", []) or []
        id_to_text = {nid: (ndata.get('text') or nid) for nid, ndata in nodes.items()}

        lines: List[str] = []
        lines.append("ESTABLISHED SESSION KNOWLEDGE:")
        lines.append("## Entity Attributes")
        # Emit compact entity attribute lines
        count = 0
        for nid, ndata in nodes.items():
            if count >= max_items:
                break
            txt = ndata.get('text') or str(nid)
            lbl = ndata.get('label') or ''
            conf = _safe_float(ndata.get('confidence', 0.0))
            if isinstance(ndata.get('attributes'), dict):
                attrs = ndata.get('attributes') or {}
                # keep only concise attributes
                keep = {k: v for k, v in attrs.items() if k in {"base", "count", "ordinal_index", "type"}}
            else:
                keep = {}
            attr_str = ", ".join(f"{k}={v}" for k, v in keep.items()) if keep else ""
            parts = [f"- {txt}"]
            if lbl:
                parts.append(f"type={lbl}")
            if attr_str:
                parts.append(attr_str)
            if conf:
                parts.append(f"confidence={conf:.2f}")
            lines.append(
                ", ".join(parts)
            )
            count += 1

        lines.append("")
        lines.append("## Relationships")
        count = 0
        for e in edges:
            if count >= max_items:
                break
            try:
                src, dst, data = e
                pred = (data or {}).get('predicate', 'related_to')
                src_text = id_to_text.get(src, str(src))
                dst_text = id_to_text.get(dst, str(dst))
                lines.append(f"- {src_text} {pred} {dst_text}")
                count += 1
            except Exception:
                continue

        lines.append("")
        lines.append("## High-Confidence Claims (>0.8)")
        count = 0
        for c in high_conf_claims or []:
            if count >= max_items:
                break
            txt = c.get('text') or ''
            conf = _safe_float(c.get('confidence', 0.0))
            if conf > 0.8 and txt:
                lines.append(f"- VERIFIED: \"{txt}\"")
                count += 1

        return "\n".join(lines)
