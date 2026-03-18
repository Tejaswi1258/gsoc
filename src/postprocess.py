# src/postprocess.py
"""
Post-processing for Spanish learner transcriptions.

Two correction layers are applied in order:

  1. Phrase corrections — fix words that Whisper splits incorrectly,
     e.g. "man zana" → "manzana".  Applied first so the result feeds
     cleanly into word-level matching.

  2. Word corrections — restore diacritics and accents that non-native
     speakers omit, e.g. "nino" → "niño".  Uses whole-word regex matching
     so "rapido" is corrected but "rapidometro" is not.

Both correction tables are loaded from config.yaml at runtime, making it
easy to extend the vocabulary without touching code.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default correction tables
# These are used when no config overrides are provided.
# ---------------------------------------------------------------------------

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
    "facil":    "fácil",
    "dificil":  "difícil",
    "util":     "útil",
    "examen":   "examen",   # already correct — kept as no-op example
}

DEFAULT_PHRASE_CORRECTIONS: dict[str, str] = {
    "man zana":  "manzana",
    "a vion":    "avión",
    "auto bus":  "autobús",
    "para guas": "paraguas",
    "bici cleta": "bicicleta",
}


# ---------------------------------------------------------------------------
# Individual correction steps
# ---------------------------------------------------------------------------

def apply_phrase_corrections(text: str, phrases: dict[str, str]) -> str:
    """
    Replace incorrectly segmented multi-word sequences with the correct form.

    Matching is case-insensitive; the corrected form is inserted as-is.
    """
    for wrong, correct in phrases.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text


def apply_word_corrections(text: str, corrections: dict[str, str]) -> str:
    """
    Restore diacritics / accents using whole-word regex matching.

    Capitalisation of the original word is preserved:
      "Nino" → "Niño",  "nino" → "niño"
    """
    if not corrections:
        return text

    pattern = r"\b(" + "|".join(re.escape(k) for k in corrections) + r")\b"

    def _replace(match: re.Match) -> str:
        original = match.group(0)
        corrected = corrections.get(original.lower(), original)
        return corrected.capitalize() if original[0].isupper() else corrected

    return re.sub(pattern, _replace, text, flags=re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase and collapse multiple spaces into one."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

def postprocess(
    text: str,
    word_corrections: Optional[dict[str, str]] = None,
    phrase_corrections: Optional[dict[str, str]] = None,
) -> str:
    """
    Run the full post-processing pipeline on a raw transcription string.

    Steps:
      1. Phrase-level corrections  (split-word merging)
      2. Word-level corrections    (accent / diacritic restoration)
      3. Normalization             (lowercase, whitespace cleanup)

    Args:
        text:               Raw transcription from Whisper.
        word_corrections:   Custom word table; falls back to DEFAULT_WORD_CORRECTIONS.
        phrase_corrections: Custom phrase table; falls back to DEFAULT_PHRASE_CORRECTIONS.

    Returns:
        Cleaned, corrected transcription string.
    """
    if not text.strip():
        return text

    word_table   = word_corrections   if word_corrections   is not None else DEFAULT_WORD_CORRECTIONS
    phrase_table = phrase_corrections if phrase_corrections is not None else DEFAULT_PHRASE_CORRECTIONS

    text = apply_phrase_corrections(text, phrase_table)
    text = apply_word_corrections(text, word_table)
    text = normalize(text)

    logger.debug("Postprocessed → '%s'", text)
    return text
