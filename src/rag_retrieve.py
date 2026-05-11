"""RAG retrieval: query FAISS indices, return top-k similar documents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rag_index import _simple_embed, _real_embed, classify_problem, parse_dafny_file

try:
    import faiss
except ImportError:
    faiss = None


INDEX_DIR = Path(__file__).resolve().parent.parent / "rag_index"


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    spec_text: str = ""
    body_text: str = ""
    invariant_text: str = ""
    full_text: str = ""
    problem_type: str = "unknown"


class RAGRetriever:
    """Retrieve similar Dafny programs from the index."""
    
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or INDEX_DIR
        self._spec_index = None
        self._body_index = None
        self._invariant_index = None
        self._documents: list[dict] = []
        self._metadata: list[dict] = []
        self._loaded = False
    
    def _load(self):
        if self._loaded:
            return
        idx = self.index_dir
        
        # Load FAISS indices or numpy arrays
        if faiss is not None:
            for name in ["spec", "body", "invariant"]:
                path = idx / f"{name}_index.faiss"
                if path.exists():
                    index = faiss.read_index(str(path))
                    setattr(self, f"_{name}_index", index)
        else:
            # Use numpy for retrieval
            for name in ["spec", "body", "invariant"]:
                path = idx / f"{name}_vectors.npy"
                if path.exists():
                    vecs = np.load(path)
                    # Build flat L2 index manually
                    if faiss is None:
                        setattr(self, f"_{name}_vectors", vecs)
        
        # Load documents
        docs_path = idx / "documents.json"
        if docs_path.exists():
            with open(docs_path) as f:
                self._documents = json.load(f)
        
        # Load metadata
        meta_path = idx / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
        
        self._loaded = True
    
    def search(
        self,
        query_text: str,
        view: str = "spec",
        k: int = 5,
        filter_type: str | None = None,
        filter_arrays: bool | None = None,
        filter_nested: bool | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[RetrievedDoc]:
        """Search the index for similar documents.
        
        Args:
            query_text: The query (spec, body, or error text)
            view: Which embedding view to search ("spec", "body", "invariant")
            k: Number of results
            filter_type: Only return docs of this problem_type
            filter_arrays: Only return docs that use arrays (True) or don't (False)
            filter_nested: Only return docs with nested loops (True) or without (False)
            exclude_ids: Doc IDs to exclude from results
        
        Returns list of RetrievedDoc sorted by score descending.
        """
        self._load()
        
        # Embed query
        query_vec = _real_embed(query_text, 384)
        query_vec = query_vec.reshape(1, -1)
        
        # Search appropriate index
        index = getattr(self, f"_{view}_index", None)
        vectors = getattr(self, f"_{view}_vectors", None)
        
        if index is not None:
            scores, indices = index.search(query_vec, min(k * 3, len(self._documents)))
        elif vectors is not None:
            # Manual cosine similarity
            similarities = np.dot(vectors, query_vec.T).flatten()
            indices = np.argsort(similarities)[::-1][:k * 3]
            scores = similarities[indices]
        else:
            return []
        
        # Build results with metadata filtering
        results = []
        exclude_ids = exclude_ids or set()
        
        for score, idx in zip(scores[0], indices[0]) if index is not None else zip(scores, indices):
            if idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            
            # Apply filters
            if exclude_ids and doc["doc_id"] in exclude_ids:
                continue
            if filter_type and doc.get("problem_type") != filter_type:
                continue
            
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            if filter_arrays is not None and meta.get("uses_arrays") != filter_arrays:
                continue
            if filter_nested is not None and meta.get("has_nested_loops") != filter_nested:
                continue
            
            results.append(RetrievedDoc(
                doc_id=doc["doc_id"],
                score=float(score),
                spec_text=doc.get("spec_text", ""),
                body_text=doc.get("body_text", ""),
                invariant_text=doc.get("invariant_text", ""),
                full_text=doc.get("full_text", ""),
                problem_type=doc.get("problem_type", "unknown"),
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_for_body_gen(
        self, spec_text: str, k: int = 3
    ) -> list[RetrievedDoc]:
        """Search optimized for body generation: find similar specs with verified bodies."""
        # Classify the query to filter by type
        methods = parse_dafny_file(spec_text)
        meta = classify_problem(methods, spec_text)
        
        results = self.search(
            spec_text,
            view="spec",
            k=k,
            filter_type=meta.get("problem_type"),
        )
        
        # If not enough results with type filter, relax
        if len(results) < k:
            more = self.search(
                spec_text,
                view="spec",
                k=k - len(results),
                filter_type=None,
                exclude_ids={r.doc_id for r in results},
            )
            results.extend(more)
        
        return results[:k]
    
    def search_for_invariants(
        self, body_text: str, ensures_text: str, k: int = 3
    ) -> list[RetrievedDoc]:
        """Search optimized for invariant filling: match loop structure."""
        query = f"{ensures_text}\n{body_text}"
        results = self.search(query, view="body", k=k)
        return results[:k]
    
    def search_for_repair(
        self, error_text: str, body_text: str, k: int = 2
    ) -> list[RetrievedDoc]:
        """Search for similar errors and their fixes."""
        query = f"{error_text[:300]}\n{body_text[:500]}"
        results = self.search(query, view="invariant", k=k)
        return results[:k]


# Global singleton
_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
    return _retriever


def format_retrieved_for_prompt(docs: list[RetrievedDoc], max_docs: int = 3) -> str:
    """Format retrieved documents as few-shot examples in a prompt."""
    if not docs:
        return ""
    
    parts = ["// ===== RETRIEVED SIMILAR VERIFIED PROGRAMS =====", ""]
    for i, doc in enumerate(docs[:max_docs]):
        parts.append(f"// --- Example {i+1} (score: {doc.score:.2f}, type: {doc.problem_type}) ---")
        if doc.full_text:
            # Truncate very long examples
            text = doc.full_text
            if len(text) > 3000:
                text = text[:3000] + "\n// ... (truncated)"
            parts.append(text)
        else:
            parts.append(f"// Spec:\n{doc.spec_text}\n")
            if doc.invariant_text and doc.invariant_text != "(no invariants)":
                parts.append(f"// Key invariants:\n{doc.invariant_text}\n")
        parts.append("")
    parts.append("// ===== END RETRIEVED EXAMPLES =====")
    return "\n".join(parts)
