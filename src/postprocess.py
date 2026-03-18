# src/postprocess.py
"""
Post-processing for Spanish learner transcriptions.

Why post-processing matters for L2 speech
------------------------------------------
Whisper transcribes what it *hears*, not what the speaker *intended*.
Non-native Spanish speakers systematically drop diacritics and accent marks
because their L1 phonology does not distinguish them.  For example:

  - English speakers say "nino" (no tilde) instead of "niño"
  - Portuguese speakers say "rapido" instead of "rápido"
  - Mandarin speakers fragment compound words: "man zana" instead of "manzana"

These are *predictable* errors — they follow patterns tied to the learner's
native language.  A rule-based post-processing layer can correct them
deterministically, improving WER without any model retraining.

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
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default correction tables
# Extend these in config.yaml — no code changes needed.
# ---------------------------------------------------------------------------

DEFAULT_WORD_CORRECTIONS: Dict[str, str] = {
    # Tilde / ñ restorations
    "nino":     "niño",
    "nina":     "niña",
    # Acute accent restorations
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
    "examen":   "examen",   # already correct — kept as a no-op example
}

# Multi-word phrases that learners (or Whisper) incorrectly segment.
# Applied before word-level corrections so merged tokens feed into
# the word table cleanly.
DEFAULT_PHRASE_CORRECTIONS: Dict[str, str] = {
    "man zana":   "manzana",
    "a vion":     "avión",
    "auto bus":   "autobús",
    "para guas":  "paraguas",
    "bici cleta": "bicicleta",
}


# ---------------------------------------------------------------------------
# Individual correction steps
# ---------------------------------------------------------------------------

def apply_phrase_corrections(text: str, phrases: Dict[str, str]) -> str:
    """
    Replace incorrectly segmented multi-word sequences with the correct form.

    Matching is case-insensitive; the corrected form is inserted as-is.

    Args:
        text:    Input transcription string.
        phrases: Mapping of wrong phrase → correct phrase.

    Returns:
        Text with all matching phrases replaced.
    """
    for wrong, correct in phrases.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text


def apply_word_corrections(text: str, corrections: Dict[str, str]) -> str:
    """
    Restore diacritics and accents using whole-word regex matching.

    Uses \\b word boundaries so partial matches are never made
    (e.g. "rapido" is fixed but "rapidometro" is left alone).

    Capitalisation of the original word is preserved:
      "Nino" → "Niño",  "nino" → "niño"

    Args:
        text:        Input transcription string.
        corrections: Mapping of unaccented word → correctly accented word.

    Returns:
        Text with diacritics restored.
    """
    if not corrections:
        return text

    pattern = r"\b(" + "|".join(re.escape(k) for k in corrections) + r")\b"

    def _replace(match: re.Match) -> str:
        original  = match.group(0)
        corrected = corrections.get(original.lower(), original)
        return corrected.capitalize() if original[0].isupper() else corrected

    return re.sub(pattern, _replace, text, flags=re.IGNORECASE)


def normalize(text: str) -> str:
    """
    Lowercase and collapse multiple whitespace characters into a single space.

    Normalisation ensures consistent comparison with reference strings during
    WER evaluation — both sides must be in the same case and spacing format.
    """
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

def postprocess(
    text: str,
    word_corrections:   Optional[Dict[str, str]] = None,
    phrase_corrections: Optional[Dict[str, str]] = None,
) -> str:
    """
    Run the full post-processing pipeline on a raw Whisper transcription.

    Steps applied in order:
      1. Phrase-level corrections  — merge incorrectly split compound words
      2. Word-level corrections    — restore missing diacritics / accents
      3. Normalisation             — lowercase + whitespace cleanup

    Args:
        text:               Raw transcription string from Whisper.
        word_corrections:   Custom word table; falls back to DEFAULT_WORD_CORRECTIONS.
        phrase_corrections: Custom phrase table; falls back to DEFAULT_PHRASE_CORRECTIONS.

    Returns:
        Cleaned, corrected, normalised transcription string.
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
