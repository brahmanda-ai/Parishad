"""
Retrieval-based checking for Parishad.

Uses retrieval to ground-check factual claims.
Implements a simple BM25/TF-IDF retriever with lazy initialization.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


# ==============================================================================
# Standalone search function (module-level, lazy-initialized singleton)
# ==============================================================================

# Module-level singleton for retriever
_retriever_instance: Optional["SimpleRetriever"] = None


def search(query: str, k: int = 5) -> list[dict]:
    """
    Search the knowledge base for relevant passages.
    
    Uses a lazy-initialized singleton retriever.
    
    Args:
        query: Search query string
        k: Number of results to return
        
    Returns:
        List of {"source_id": str, "snippet": str, "score": float}
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = SimpleRetriever()
    
    return _retriever_instance.search(query, k=k)


def load_corpus(corpus_path: str) -> None:
    """
    Load a corpus into the singleton retriever.
    
    Args:
        corpus_path: Path to JSONL corpus file
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = SimpleRetriever()
    
    _retriever_instance.load_corpus(corpus_path)


def reset_retriever() -> None:
    """Reset the singleton retriever (useful for testing)."""
    global _retriever_instance
    _retriever_instance = None


# ==============================================================================
# Simple TF-IDF/BM25 Retriever Implementation
# ==============================================================================

class SimpleRetriever:
    """
    Simple BM25-based retriever for fact checking.
    
    Uses TF-IDF with BM25 scoring. Initialized lazily and cached.
    For production, consider using:
    - rank_bm25 library
    - Whoosh for full-text search
    - FAISS for vector similarity
    """
    
    # BM25 parameters
    K1 = 1.5
    B = 0.75
    
    def __init__(self):
        """Initialize empty retriever."""
        self._documents: list[dict] = []
        self._doc_freqs: Counter = Counter()
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._tokenized_docs: list[list[str]] = []
        self._initialized: bool = False
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Remove very short tokens and stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'while', 'although', 'this', 'that', 'these',
                     'those', 'it', 'its'}
        return [t for t in tokens if len(t) > 1 and t not in stopwords]
    
    def load_corpus(self, corpus_path: str) -> None:
        """
        Load documents from a JSONL file.
        
        Expected format per line:
        {"id": "doc_id", "text": "document text", "source": "optional source"}
        
        Args:
            corpus_path: Path to JSONL file
        """
        path = Path(corpus_path)
        if not path.exists():
            logger.warning(f"Corpus file not found: {corpus_path}")
            return
        
        self._documents = []
        self._tokenized_docs = []
        self._doc_lengths = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    text = doc.get('text', '')
                    tokens = self._tokenize(text)
                    
                    self._documents.append({
                        'id': doc.get('id', str(len(self._documents))),
                        'text': text,
                        'source': doc.get('source', 'corpus')
                    })
                    self._tokenized_docs.append(tokens)
                    self._doc_lengths.append(len(tokens))
                    
                    # Update document frequency
                    for token in set(tokens):
                        self._doc_freqs[token] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        
        self._initialized = True
        logger.info(f"Loaded {len(self._documents)} documents from {corpus_path}")
    
    def add_documents(self, documents: list[dict]) -> None:
        """
        Add documents directly (without file loading).
        
        Args:
            documents: List of {"id": str, "text": str, "source": str}
        """
        for doc in documents:
            text = doc.get('text', '')
            tokens = self._tokenize(text)
            
            self._documents.append({
                'id': doc.get('id', str(len(self._documents))),
                'text': text,
                'source': doc.get('source', 'added')
            })
            self._tokenized_docs.append(tokens)
            self._doc_lengths.append(len(tokens))
            
            for token in set(tokens):
                self._doc_freqs[token] += 1
        
        if self._doc_lengths:
            self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths)
        
        self._initialized = bool(self._documents)
    
    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc_tokens = self._tokenized_docs[doc_idx]
        doc_len = self._doc_lengths[doc_idx]
        n_docs = len(self._documents)
        
        if n_docs == 0 or self._avg_doc_length == 0:
            return 0.0
        
        score = 0.0
        doc_tf = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in doc_tf:
                continue
            
            tf = doc_tf[token]
            df = self._doc_freqs.get(token, 0)
            
            # IDF component
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            
            # BM25 TF component
            tf_component = (tf * (self.K1 + 1)) / (
                tf + self.K1 * (1 - self.B + self.B * doc_len / self._avg_doc_length)
            )
            
            score += idf * tf_component
        
        return score
    
    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of {"source_id": str, "snippet": str, "score": float}
        """
        if not self._initialized or not self._documents:
            # Return empty results if no corpus loaded
            return []
        
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Score all documents
        scores = []
        for i in range(len(self._documents)):
            score = self._bm25_score(query_tokens, i)
            if score > 0:
                scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for doc_idx, score in scores[:k]:
            doc = self._documents[doc_idx]
            # Create snippet (first 200 chars)
            snippet = doc['text'][:200]
            if len(doc['text']) > 200:
                snippet += "..."
            
            results.append({
                'source_id': doc['id'],
                'snippet': snippet,
                'score': round(score, 4)
            })
        
        return results


# ==============================================================================
# Dataclasses for structured results
# ==============================================================================


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""
    
    query: str
    passages: list[dict[str, Any]]
    source: str
    confidence: float = 0.0


@dataclass
class FactCheckResult:
    """Result from fact-checking a claim."""
    
    claim: str
    supported: Optional[bool]  # None = unknown
    evidence: list[str]
    confidence: float
    explanation: str


class RetrievalChecker:
    """
    Retrieval-based fact checker.
    
    Uses SimpleRetriever (BM25) for retrieval and provides
    claim extraction and fact-checking capabilities.
    """
    
    def __init__(
        self,
        knowledge_base_path: Optional[str] = None,
        top_k: int = 5,
    ):
        """
        Initialize retrieval checker.
        
        Args:
            knowledge_base_path: Path to local knowledge base (JSONL)
            top_k: Number of passages to retrieve
        """
        self.top_k = top_k
        self._retriever = SimpleRetriever()
        
        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)
    
    def load_knowledge_base(self, path: str) -> None:
        """
        Load a knowledge base from disk.
        
        Args:
            path: Path to JSONL corpus file
        """
        self._retriever.load_corpus(path)
    
    def add_documents(self, documents: list[dict]) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of {"id": str, "text": str, "source": str}
        """
        self._retriever.add_documents(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: Query to search for
            top_k: Override default top_k
            
        Returns:
            RetrievalResult with matched passages
        """
        k = top_k or self.top_k
        results = self._retriever.search(query, k=k)
        
        # Convert to passages format
        passages = []
        for r in results:
            passages.append({
                "text": r["snippet"],
                "source": r["source_id"],
                "score": r["score"]
            })
        
        # Calculate confidence based on top score
        confidence = 0.0
        if results:
            # Normalize score to 0-1 range (heuristic)
            top_score = results[0]["score"]
            confidence = min(1.0, top_score / 10.0)
        
        return RetrievalResult(
            query=query,
            passages=passages,
            source="bm25",
            confidence=confidence,
        )
    
    def extract_claims(self, text: str) -> list[str]:
        """
        Extract factual claims from text.
        
        Uses simple heuristics to identify claim-like sentences.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of extracted claims
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Skip short sentences
            if len(sentence) < 20:
                continue
            
            # Skip questions
            if sentence.endswith('?'):
                continue
            
            # Skip meta-sentences (instructions, etc.)
            meta_patterns = [
                r'^(note|disclaimer|warning|important):',
                r'^(first|second|third|finally|however|therefore)',
                r'^(let me|i will|we can|you should)',
            ]
            is_meta = any(re.match(p, sentence.lower()) for p in meta_patterns)
            if is_meta:
                continue
            
            claims.append(sentence)
        
        return claims
    
    def check_claim(self, claim: str) -> FactCheckResult:
        """
        Check a single factual claim.
        
        Retrieves relevant passages and checks for support.
        
        Args:
            claim: Claim to verify
            
        Returns:
            FactCheckResult with verification status
        """
        # Retrieve relevant passages
        result = self.retrieve(claim, top_k=3)
        
        if not result.passages:
            return FactCheckResult(
                claim=claim,
                supported=None,  # Unknown - no evidence
                evidence=[],
                confidence=0.0,
                explanation="No relevant passages found"
            )
        
        # Simple heuristic: check if claim terms appear in passages
        claim_tokens = set(self._retriever._tokenize(claim))
        
        evidence = []
        support_scores = []
        
        for passage in result.passages:
            passage_text = passage.get("text", "")
            passage_tokens = set(self._retriever._tokenize(passage_text))
            
            # Calculate overlap
            if claim_tokens:
                overlap = len(claim_tokens & passage_tokens) / len(claim_tokens)
            else:
                overlap = 0.0
            
            support_scores.append(overlap)
            if overlap > 0.3:
                evidence.append(passage_text[:100])
        
        # Determine support
        avg_overlap = sum(support_scores) / len(support_scores) if support_scores else 0
        
        if avg_overlap > 0.5:
            supported = True
            explanation = "Claim terms found in retrieved passages"
        elif avg_overlap > 0.2:
            supported = None  # Uncertain
            explanation = "Partial overlap with retrieved passages"
        else:
            supported = None  # Unknown, not necessarily false
            explanation = "Low overlap with retrieved passages"
        
        return FactCheckResult(
            claim=claim,
            supported=supported,
            evidence=evidence[:3],  # Limit evidence
            confidence=min(avg_overlap, result.confidence),
            explanation=explanation
        )
    
    def check_all_claims(
        self,
        text: str,
        max_claims: int = 10,
    ) -> list[FactCheckResult]:
        """
        Extract and check all claims in text.
        
        Args:
            text: Text to check
            max_claims: Maximum claims to check
            
        Returns:
            List of fact check results
        """
        claims = self.extract_claims(text)[:max_claims]
        return [self.check_claim(claim) for claim in claims]
    
    def get_grounding_context(
        self,
        query: str,
        max_tokens: int = 500,
    ) -> str:
        """
        Get grounding context for a query.
        
        Retrieves relevant passages and formats them
        as context for LLM verification.
        
        Args:
            query: Query to ground
            max_tokens: Approximate max context length
            
        Returns:
            Formatted context string
        """
        result = self.retrieve(query)
        
        if not result.passages:
            return ""
        
        context_parts = []
        total_length = 0
        
        for passage in result.passages:
            text = passage.get("text", "")
            source = passage.get("source", "unknown")
            
            if total_length + len(text) > max_tokens * 4:  # Rough char estimate
                break
            
            context_parts.append(f"[Source: {source}]\n{text}")
            total_length += len(text)
        
        return "\n\n".join(context_parts)
