"""
Centralized degenerate output detection and sanitization for Parishad.

Small language models (0.5-3B) sometimes enter repetition loops, producing
output like:
    "The final answer is 'The final answer is 'The final answer is..."

This module provides robust detection and sanitization that works across
ALL role outputs, preventing garbage from reaching the user.
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Detection
# ============================================================================

def detect_degenerate_output(text: str, min_length: int = 60) -> bool:
    """
    Detect if text is degenerate (repetitive/looping model output).
    
    Uses multiple heuristics to catch various forms of repetition:
    1. Repetitive n-gram dominance
    2. Low unique-word ratio
    3. Known degenerate phrase patterns
    
    Args:
        text: The model output to check
        min_length: Minimum text length to consider checking (short text
                    is rarely degenerate and could false-positive)
    
    Returns:
        True if the text appears degenerate
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    text = text.strip()
    
    # --- Heuristic 1: Known recursive phrase patterns ---
    if _has_recursive_phrases(text):
        return True
    
    # --- Heuristic 2: N-gram repetition dominance ---
    if _has_ngram_dominance(text):
        return True
    
    # --- Heuristic 3: Very low unique-word ratio ---
    if _has_low_unique_ratio(text):
        return True
    
    return False


def _has_recursive_phrases(text: str) -> bool:
    """
    Check for known recursive/looping phrase patterns from small models.
    
    Detects patterns like:
    - "The final answer is 'The final answer is..."
    - "Answer: Answer: Answer:..."
    - "The answer is: The answer is:..."
    """
    lower = text.lower()
    
    # Pattern: same phrase repeated 3+ times in sequence
    # This catches "The final answer is" repeated, "Answer:" repeated, etc.
    recursive_patterns = [
        r'(the\s+final\s+answer\s+is[:\s]*){3,}',
        r'(final\s+answer[:\s]*){3,}',
        r'(the\s+answer\s+is[:\s]*){3,}',
        r'(answer\s*:\s*){3,}',
        r'(response\s*:\s*){3,}',
        r'(result\s*:\s*){3,}',
        r'(output\s*:\s*){3,}',
    ]
    
    for pattern in recursive_patterns:
        if re.search(pattern, lower):
            return True
    
    # Generic pattern: any phrase of 3-8 words repeated 4+ times
    # This catches novel repetition patterns we haven't seen yet
    words = lower.split()
    if len(words) >= 12:
        for ngram_size in range(3, 9):
            ngrams = []
            for i in range(len(words) - ngram_size + 1):
                ngram = ' '.join(words[i:i + ngram_size])
                ngrams.append(ngram)
            
            if not ngrams:
                continue
            
            counter = Counter(ngrams)
            most_common_ngram, most_common_count = counter.most_common(1)[0]
            
            # If a single n-gram appears 4+ times AND accounts for >30% of text
            if most_common_count >= 4:
                ngram_word_coverage = (most_common_count * ngram_size) / len(words)
                if ngram_word_coverage > 0.30:
                    logger.debug(
                        f"Recursive n-gram detected: '{most_common_ngram}' "
                        f"appears {most_common_count}x, coverage={ngram_word_coverage:.0%}"
                    )
                    return True
    
    return False


def _has_ngram_dominance(text: str) -> bool:
    """
    Check if a single n-gram dominates the text (>40% of all n-grams).
    
    This catches repetitive text even if it's not a known phrase.
    """
    words = text.lower().split()
    
    if len(words) < 15:
        return False
    
    # Check bigrams and trigrams
    for n in (2, 3):
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i + n]))
        
        if not ngrams:
            continue
        
        counter = Counter(ngrams)
        total = len(ngrams)
        _, top_count = counter.most_common(1)[0]
        
        ratio = top_count / total
        if ratio > 0.40:
            return True
    
    return False


def _has_low_unique_ratio(text: str) -> bool:
    """
    Check if the text has an unusually low ratio of unique words.
    
    Normal text has ~40-70% unique words. Degenerate text often has <15%.
    """
    words = text.lower().split()
    
    if len(words) < 20:
        return False
    
    unique_count = len(set(words))
    ratio = unique_count / len(words)
    
    # Very low unique ratio strongly suggests repetition
    return ratio < 0.12


# ============================================================================
# Sanitization
# ============================================================================

def sanitize_output(text: str) -> str:
    """
    Clean degenerate text by stripping repetitive prefixes and wrappers.
    
    Tries to extract any meaningful content buried within the repetition.
    If no meaningful content is found, returns empty string so the caller
    can use its own fallback.
    
    Args:
        text: The degenerate text to sanitize
        
    Returns:
        Cleaned text, or empty string if no meaningful content can be recovered
    """
    if not text:
        return ""
    
    original_text = text
    
    # Step 1: Strip all recursive "The final answer is" / "Answer:" prefixes
    text = _strip_recursive_prefixes(text)
    
    # Step 2: Strip leftover quote nesting
    text = _strip_nested_quotes(text)
    
    # Step 3: Check if what remains is meaningful
    cleaned = text.strip()
    
    # Reject very short fragments — they're truncation remnants, not real answers
    if cleaned and len(cleaned) < 10:
        cleaned = ""
    
    if cleaned and not detect_degenerate_output(cleaned, min_length=30):
        # We recovered something meaningful
        return cleaned
    
    # Step 4: Try to extract meaningful content from the original text
    meaningful = extract_meaningful_content(original_text)
    if meaningful:
        return meaningful
    
    # Nothing salvageable
    return ""


def _strip_recursive_prefixes(text: str) -> str:
    """
    Aggressively strip all known repetitive prefix patterns.
    
    Handles:
    - The final answer is: 'The final answer is: ...'
    - "The final answer is: "The final answer is:..."
    - Mixed quotes, colons, whitespace variations
    - <think>...</think> tags from reasoning models
    """
    # Strip <think>...</think> tags and their content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip orphaned opening/closing think tags
    text = re.sub(r'</?think>', '', text)
    text = text.strip()
    
    # Do up to 50 iterations to peel away nesting
    for _ in range(50):
        before = text
        text = text.strip()
        
        # Strip leading quotes
        while text and text[0] in '"\'':
            text = text[1:]
        
        # Strip trailing quotes
        while text and text[-1] in '"\'':
            text = text[:-1]
        
        # Strip known prefix patterns (case-insensitive)
        prefix_patterns = [
            r'^the\s+final\s+answer\s+is\s*:?\s*',
            r'^final\s+answer\s*:?\s*',
            r'^the\s+answer\s+is\s*:?\s*',
            r'^answer\s*:?\s*',
            r'^response\s*:?\s*',
            r'^result\s*:?\s*',
            r'^output\s*:?\s*',
        ]
        
        for pattern in prefix_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # If nothing changed, stop
        if text == before:
            break
    
    return text.strip()


def _strip_nested_quotes(text: str) -> str:
    """Strip layers of nested quotes that small models produce."""
    for _ in range(20):
        before = text.strip()
        
        # Remove matching outer quotes
        if len(before) >= 2:
            if (before[0] == '"' and before[-1] == '"') or \
               (before[0] == "'" and before[-1] == "'"):
                before = before[1:-1]
        
        if before == text.strip():
            break
        text = before
    
    return text.strip()


def extract_meaningful_content(text: str) -> str:
    """
    Try to extract meaningful content from within degenerate text.
    
    Strategies:
    1. Find the longest non-repeating substring
    2. Look for sentence-like content between repetition markers
    
    Args:
        text: Degenerate text to extract from
        
    Returns:
        Meaningful content if found, empty string otherwise
    """
    if not text:
        return ""
    
    # Strategy 1: Look for content after "is:" or "is " that isn't more repetition
    # Sometimes the actual answer is buried: "The final answer is: AI is... The final answer is..."
    lower = text.lower()
    
    # Find all positions where "the final answer is" appears
    marker = "the final answer is"
    positions = []
    start = 0
    while True:
        pos = lower.find(marker, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + len(marker)
    
    if len(positions) >= 2:
        # Get the text between the first marker and the second
        after_first = text[positions[0] + len(marker):]
        # Strip leading colon/quotes/whitespace
        after_first = re.sub(r'^[\s:"\']+ ', '', after_first)
        
        # Get text up to the next occurrence of the marker
        next_marker = lower.find(marker, positions[0] + len(marker))
        if next_marker > positions[0] + len(marker):
            between = text[positions[0] + len(marker):next_marker].strip()
            between = re.sub(r'^[\s:"\']+ ', '', between)
            between = between.rstrip('"\'')
            
            if len(between) > 20 and not detect_degenerate_output(between, min_length=15):
                return between.strip()
    
    # Strategy 2: Find sentences that look like real content
    sentences = re.findall(r'[A-Z][^.!?]{15,}[.!?]', text)
    if sentences:
        # Filter out sentences that are just repetition markers
        real_sentences = [
            s for s in sentences
            if 'final answer' not in s.lower() 
            and 'the answer is' not in s.lower()
        ]
        if real_sentences:
            return ' '.join(real_sentences[:3])
    
    return ""
