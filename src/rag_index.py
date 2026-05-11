"""RAG index builder: parse Dafny programs, embed them, build FAISS indices."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


# ---------------------------------------------------------------------------
# Dafny parser (regex-based, not a full parser)
# ---------------------------------------------------------------------------

_METHOD_RE = re.compile(
    r'(?:method|lemma)\s+(\w+)\s*(?:<[^>]*>)?\s*\((.*?)\)\s*'
    r'(?:returns\s*\((.*?)\))?',
    re.DOTALL,
)

_REQUIRES_RE = re.compile(r'requires\s+(.+?)(?=\n\s*requires|\n\s*ensures|\n\s*reads|\n\s*modifies|\n\s*decreases|\n\s*\{|\n\s*$)', re.DOTALL)

_ENSURES_RE = re.compile(r'ensures\s+(.+?)(?=\n\s*ensures|\n\s*reads|\n\s*modifies|\n\s*decreases|\n\s*\{|\n\s*$)', re.DOTALL)

_WHILE_RE = re.compile(
    r'while\s+(.+?)\n((?:\s*invariant\s+[^\n]+\n)*)\s*'
    r'(?:decreases\s+([^\n]+)\n)?',
    re.DOTALL,
)

_INVARIANT_RE = re.compile(r'invariant\s+(.+?)(?=\n\s*invariant|\n\s*decreases|\n\s*\{|\n\s*//|\n\s*$)', re.DOTALL)

_PREDICATE_RE = re.compile(
    r'(?:ghost\s+)?predicate\s+(\w+)\s*(?:<[^>]*>)?\s*\((.*?)\)',
    re.DOTALL,
)

_FUNCTION_RE = re.compile(
    r'(?:ghost\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\((.*?)\)\s*:\s*(\w+)',
    re.DOTALL,
)


@dataclass
class LoopInfo:
    condition: str
    invariants: list[str] = field(default_factory=list)
    decreases: str = ""


@dataclass
class MethodInfo:
    name: str
    params: str
    returns: str
    requires: list[str] = field(default_factory=list)
    ensures: list[str] = field(default_factory=list)
    body: str = ""
    loops: list[LoopInfo] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)


@dataclass
class IndexedDoc:
    doc_id: str
    source_file: str
    methods: list[MethodInfo] = field(default_factory=list)
    
    # Embedding views
    spec_text: str = ""
    body_text: str = ""
    invariant_text: str = ""
    full_text: str = ""
    
    spec_embedding: np.ndarray | None = None
    body_embedding: np.ndarray | None = None
    invariant_embedding: np.ndarray | None = None
    
    # Metadata
    problem_type: str = "unknown"
    num_loops: int = 0
    has_nested_loops: bool = False
    uses_arrays: bool = False
    uses_seqs: bool = False
    uses_sets: bool = False
    uses_recursion: bool = False
    max_loop_depth: int = 0


def parse_dafny_file(content: str) -> list[MethodInfo]:
    """Extract methods, predicates, and their specs from Dafny source."""
    methods = []
    
    # Strip comments
    content_no_comments = re.sub(r'//[^\n]*', '', content)
    content_no_comments = re.sub(r'/\*.*?\*/', '', content_no_comments, flags=re.DOTALL)
    
    # Find method/lemma blocks
    for m in _METHOD_RE.finditer(content_no_comments):
        name = m.group(1)
        params = m.group(2).strip()
        returns = (m.group(3) or "").strip()
        
        # Find the body (text between the first { after method sig and matching })
        sig_end = m.end()
        body_start = content_no_comments.find('{', sig_end)
        if body_start == -1:
            continue
        body_end = _find_matching_brace(content_no_comments, body_start)
        if body_end == -1:
            continue
        body = content_no_comments[body_start + 1:body_end].strip()
        surrounding = content_no_comments[m.start():body_end + 1]
        
        # Extract requires/ensures
        requires = _REQUIRES_RE.findall(surrounding)
        ensures = _ENSURES_RE.findall(surrounding)
        
        # Clean up requires/ensures
        requires = [r.strip().rstrip(';') for r in requires if r.strip()]
        ensures = [e.strip().rstrip(';') for e in ensures if e.strip()]
        
        # Extract loops and invariants
        loops = []
        while_positions = [(m2.start(), m2.end()) for m2 in re.finditer(r'while\s+', body)]
        for wp_start, _ in while_positions:
            cond_match = re.match(r'while\s+(.+?)\n', body[wp_start:], re.DOTALL)
            if not cond_match:
                continue
            condition = cond_match.group(1).strip()
            
            # Find the while body
            while_body_start = body.find('{', wp_start)
            if while_body_start == -1:
                continue
            while_body_end = _find_matching_brace(body, while_body_start)
            while_body_text = body[while_body_start:while_body_end + 1]
            
            # Extract invariants from the while header (between while cond and {)
            header = body[wp_start:while_body_start]
            inv_matches = _INVARIANT_RE.findall(header)
            invariants = [inv.strip() for inv in inv_matches if inv.strip()]
            
            # Extract decreases
            dec_match = re.search(r'decreases\s+([^\n{]+)', header)
            decreases = dec_match.group(1).strip() if dec_match else ""
            
            loops.append(LoopInfo(
                condition=condition,
                invariants=invariants,
                decreases=decreases,
            ))
        
        method = MethodInfo(
            name=name,
            params=params,
            returns=returns,
            requires=requires,
            ensures=ensures,
            body=body,
            loops=loops,
        )
        methods.append(method)
    
    return methods


def _find_matching_brace(text: str, start: int) -> int:
    """Find the matching closing brace for an opening brace at position start."""
    if start >= len(text) or text[start] != '{':
        return -1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


# ---------------------------------------------------------------------------
# Metadata classifier
# ---------------------------------------------------------------------------

def classify_problem(methods: list[MethodInfo], full_text: str) -> dict:
    """Classify problem type and extract metadata."""
    text_lower = full_text.lower()
    
    # Count loops
    total_loops = sum(len(m.loops) for m in methods)
    has_nested = "while" in text_lower and text_lower.count("while") > 1
    
    # Check for nested loops (while inside while)
    nested = False
    for m in methods:
        for loop in m.loops:
            loop_body_start = m.body.find('{', m.body.find(loop.condition))
            if loop_body_start != -1:
                loop_body = m.body[loop_body_start:]
                if 'while' in loop_body:
                    nested = True
                    break
    
    # Problem type
    if "binary" in text_lower and "search" in text_lower:
        ptype = "binary_search"
    elif "sort" in text_lower or "bubble" in text_lower or "insertion" in text_lower:
        ptype = "sorting"
    elif "recursive" in text_lower or "fibonacci" in text_lower or "factorial" in text_lower:
        ptype = "recursive"
    elif nested:
        ptype = "nested_loop"
    elif "prime" in text_lower or "gcd" in text_lower or "sqrt" in text_lower or "mod" in text_lower:
        ptype = "math"
    elif total_loops >= 1:
        ptype = "linear_scan"
    else:
        ptype = "simple"
    
    return {
        "problem_type": ptype,
        "num_loops": total_loops,
        "has_nested_loops": nested,
        "uses_arrays": "array" in text_lower,
        "uses_seqs": "seq<" in text_lower,
        "uses_sets": "set<" in text_lower,
        "uses_recursion": any(m.name.lower() in text_lower.lower().replace(m.name, "") for m in methods if m.name in full_text),
        "max_loop_depth": 2 if nested else (1 if total_loops > 0 else 0),
    }


# ---------------------------------------------------------------------------
# Embedding (using API or local model)
# ---------------------------------------------------------------------------

# Simple local embedding using weighted keyword matching as fallback
# when no embedding API is available

# Sentence-transformer embedding (replaces _simple_embed when available)
_EMBED_MODEL = None
_EMBED_DIM = 384

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            _EMBED_MODEL = False
    return _EMBED_MODEL

def _real_embed(text: str, dim: int = 384) -> np.ndarray:
    """Embed text using sentence-transformers, fall back to keyword embedding."""
    model = _get_embed_model()
    if model and model is not False:
        vec = model.encode([text], show_progress_bar=False)[0]
        vec = vec / np.linalg.norm(vec)
        return vec.astype(np.float32)
    return _simple_embed(text, dim)


def _simple_embed(text: str, dim: int = 256) -> np.ndarray:
    """Simple bag-of-keywords embedding as fallback."""
    # Dafny-specific keywords
    keywords = [
        "method", "function", "predicate", "lemma", "returns",
        "requires", "ensures", "reads", "modifies", "decreases",
        "invariant", "assert", "assume", "ghost", "calc",
        "while", "for", "if", "else", "match", "var",
        "array", "seq", "set", "map", "multiset", "iset",
        "int", "nat", "bool", "real", "char", "string",
        "forall", "exists", "old", "fresh", "allocated",
        "sorted", "search", "sort", "max", "min", "sum",
        "count", "find", "insert", "delete", "reverse",
        "Length", "binary", "linear", "loop", "recursive",
        "0", "1", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "==>",
    ]
    
    vec = np.zeros(dim, dtype=np.float32)
    text_lower = text.lower()
    
    for i, kw in enumerate(keywords):
        count = text_lower.count(kw.lower())
        idx = i % dim
        vec[idx] += count * 0.1
    
    # Add n-gram features for Dafny patterns
    patterns = [
        r'forall\s+\w+', r'ensures\s+', r'requires\s+',
        r'invariant\s+', r'decreases\s+', r'while\s+',
        r'a\[', r'\|[a-z]+\|',
    ]
    for i, pat in enumerate(patterns):
        count = len(re.findall(pat, text_lower))
        idx = (len(keywords) + i) % dim
        vec[idx] += count * 0.2
    
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    
    return vec.astype(np.float32)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    ground_truth_dir: Path,
    output_dir: Path,
    embedding_dim: int = 256,
) -> list[IndexedDoc]:
    """Build FAISS indices from DafnyBench ground truth files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    docs = []
    
    for dfy_file in sorted(ground_truth_dir.glob("*.dfy")):
        try:
            content = dfy_file.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        
        if len(content) < 20:
            continue
        
        # Parse
        methods = parse_dafny_file(content)
        if not methods:
            continue
        
        # Classify
        meta = classify_problem(methods, content)
        
        # Build text views
        spec_parts = []
        body_parts = []
        inv_parts = []
        
        for m in methods:
            spec_parts.append(f"method {m.name}({m.params}) returns ({m.returns})")
            for r in m.requires:
                spec_parts.append(f"  requires {r}")
            for e in m.ensures:
                spec_parts.append(f"  ensures {e}")
            
            body_parts.append(m.body)
            
            for loop in m.loops:
                for inv in loop.invariants:
                    inv_parts.append(f"invariant {inv}")
                if loop.decreases:
                    inv_parts.append(f"decreases {loop.decreases}")
        
        spec_text = "\n".join(spec_parts)
        body_text = "\n".join(body_parts)
        invariant_text = "\n".join(inv_parts) if inv_parts else "(no invariants)"
        
        doc_id = dfy_file.stem[:80]
        
        doc = IndexedDoc(
            doc_id=doc_id,
            source_file=str(dfy_file),
            methods=methods,
            spec_text=spec_text,
            body_text=body_text,
            invariant_text=invariant_text,
            full_text=content,
            **meta,
        )
        
        # Generate embeddings
        doc.spec_embedding = _real_embed(spec_text, embedding_dim)
        doc.body_embedding = _real_embed(body_text, embedding_dim)
        doc.invariant_embedding = _real_embed(invariant_text, embedding_dim)
        
        docs.append(doc)
    
    if not docs:
        print("No documents parsed!", file=sys.stderr)
        return []
    
    # Build FAISS indices
    spec_vecs = np.array([d.spec_embedding for d in docs], dtype=np.float32)
    body_vecs = np.array([d.body_embedding for d in docs], dtype=np.float32)
    inv_vecs = np.array([d.invariant_embedding for d in docs], dtype=np.float32)
    
    if faiss is not None:
        for name, vecs in [("spec", spec_vecs), ("body", body_vecs), ("invariant", inv_vecs)]:
            index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine for normalized vectors)
            index.add(vecs)
            faiss.write_index(index, str(output_dir / f"{name}_index.faiss"))
    else:
        print("FAISS not available, saving as numpy arrays only", file=sys.stderr)
        np.save(output_dir / "spec_vectors.npy", spec_vecs)
        np.save(output_dir / "body_vectors.npy", body_vecs)
        np.save(output_dir / "invariant_vectors.npy", inv_vecs)
    
    # Save metadata and documents
    metadata = []
    for d in docs:
        metadata.append({
            "doc_id": d.doc_id,
            "source_file": d.source_file,
            "problem_type": d.problem_type,
            "num_loops": d.num_loops,
            "has_nested_loops": d.has_nested_loops,
            "uses_arrays": d.uses_arrays,
            "uses_seqs": d.uses_seqs,
            "uses_sets": d.uses_sets,
            "uses_recursion": d.uses_recursion,
            "spec_text": d.spec_text[:500],
            "invariant_text": d.invariant_text[:500],
        })
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save full docs for retrieval
    docs_data = []
    for d in docs:
        docs_data.append({
            "doc_id": d.doc_id,
            "source_file": d.source_file,
            "spec_text": d.spec_text,
            "body_text": d.body_text,
            "invariant_text": d.invariant_text,
            "full_text": d.full_text,
            "problem_type": d.problem_type,
        })
    
    with open(output_dir / "documents.json", "w") as f:
        json.dump(docs_data, f, indent=2)
    
    return docs


if __name__ == "__main__":
    # Quick test
    import sys
    ground = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/DafnyBench/DafnyBench/dataset/ground_truth")
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/researcher/Project/test_dafny/rag_index")
    
    docs = build_index(ground, output)
    print(f"Indexed {len(docs)} documents")
    
    # Show type distribution
    types = {}
    for d in docs:
        types[d.problem_type] = types.get(d.problem_type, 0) + 1
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")
