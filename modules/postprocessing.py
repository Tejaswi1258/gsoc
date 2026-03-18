# modules/postprocessing.py
"""
Post-processing pipeline for Spanish learner transcriptions.
Applies accent restoration and multi-word corrections loaded from config.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default corrections — can be overridden via config.yaml
DEFAULT_WORD_CORRECTIONS: dict[str, str] = {
    "nino":     "niño",
    "nina":     "niña",
    "rapido":   "rápido",
    "rapida":   "rápida",
    "senor":    "señor",
    "senora":   "señora",
    "arbol":    "árbol",
    "musica":   "música",
    "telefono": "teléfono",
    "numero":   "número",
    "pagina":   "página",
    "cafe":     "café",
    "ingles":   "inglés",
    "frances":  "francés",
}

# Multi-word phrase corrections (applied before word-level corrections)
DEFAULT_PHRASE_CORRECTIONS: dict[str, str] = {
    "man zana":   "manzana",
    "a vion":     "avión",
    "auto bus":   "autobús",
    "para guas":  "paraguas",
}


def apply_phrase_corrections(text: str, phrases: dict[str, str]) -> str:
    """Replace incorrectly split multi-word phrases."""
    for wrong, correct in phrases.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text


def apply_word_corrections(text: str, corrections: dict[str, str]) -> str:
    """Replace individual words using whole-word matching (case-insensitive)."""
    def replace(match):
        word = match.group(0)
        corrected = corrections.get(word.lower())
        if corrected is None:
            return word
        # Preserve original capitalisation
        return corrected.capitalize() if word[0].isupper() else corrected

    pattern = r'\b(' + '|'.join(re.escape(k) for k in corrections) + r')\b'
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Lowercase, strip extra whitespace."""
    return re.sub(r'\s+', ' ', text.strip().lower())


def postprocess_text(
    text: str,
    word_corrections: Optional[dict[str, str]] = None,
    phrase_corrections: Optional[dict[str, str]] = None,
) -> str:
    """
    Full post-processing pipeline:
      1. Phrase-level corrections (split words)
      2. Word-level accent/diacritic restoration
      3. Normalization

    Args:
        text: Raw transcription string
        word_corrections: Override default word corrections dict
        phrase_corrections: Override default phrase corrections dict

    Returns:
        Cleaned and corrected transcription string
    """
    if not text.strip():
        return text

    words = word_corrections or DEFAULT_WORD_CORRECTIONS
    phrases = phrase_corrections or DEFAULT_PHRASE_CORRECTIONS

    text = apply_phrase_corrections(text, phrases)
    text = apply_word_corrections(text, words)
    text = normalize_text(text)

    logger.debug("Postprocessed text: %s", text)
    return text
