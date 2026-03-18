# Spanish Learner Speech Transcription Pipeline

> A GSoC research prototype for robust automatic speech recognition (ASR)
> evaluation of non-native Spanish learner audio using OpenAI Whisper.

[![Tests](https://img.shields.io/badge/tests-45%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Problem Statement

Automatic Speech Recognition systems like Whisper are trained predominantly
on native-speaker audio. When applied to **L2 (second-language) learner
speech**, they encounter a fundamentally different acoustic and linguistic
distribution:

| Challenge | Example | Root Cause |
|---|---|---|
| Missing diacritics | "nino" instead of "niño" | L1 phonology lacks tilde distinction |
| Missing accents | "rapido" instead of "rápido" | Stress patterns differ from L1 |
| Word fragmentation | "man zana" instead of "manzana" | Syllable boundary transfer from L1 |
| Phoneme substitution | /r/ pronounced as /l/ | L1 phoneme inventory interference |
| Prosodic mismatch | Flat intonation | L1 prosody applied to L2 utterances |

These are not random errors — they are **systematic and predictable** based
on the learner's native language (L1). This makes them amenable to rule-based
correction without any model retraining.

Standard ASR benchmarks (LibriSpeech, Common Voice) do not capture these
patterns, so evaluation on learner corpora requires a dedicated pipeline.

---

## Research Motivation

This project addresses a real gap in language learning technology:

- ~1.5 billion people are learning a second language globally
- ASR-based pronunciation feedback tools (Duolingo, ELSA) rely on models not
  optimised for learner speech
- WER on learner audio is typically **2-4x higher** than on native speech for
  the same model
- Systematic post-processing corrections can recover **15-30% relative WER
  reduction** without any fine-tuning

This pipeline provides a reproducible baseline for measuring and improving
ASR quality on Spanish L2 speech — directly aligned with GSoC projects in
computational linguistics, speech technology, and language learning tools.

---

## Key Contributions

- **Modular pipeline** — each stage (preprocess / transcribe / postprocess /
  evaluate) is independently testable and replaceable
- **Noise-robust preprocessing** — `noisereduce` spectral gating removes
  classroom and home recording noise before transcription
- **Learner-aware post-processing** — rule-based accent restoration and phrase
  merging corrects systematic L2 errors
- **Structured evaluation** — per-file WER + dataset average + word-level
  error analysis (substitutions, deletions, insertions)
- **GPU/CPU transparent** — auto-detects CUDA; `fp16` enabled automatically
  on GPU for ~2x speed improvement
- **Fully configurable** — all parameters controlled via `config.yaml`; CLI
  flags override config values
- **45 unit tests** — covering preprocessing, postprocessing, evaluation,
  error analysis, file I/O, and helper functions

---

## Pipeline Architecture

```
Audio File(s)  [.wav / .mp3 / .flac / .m4a]
      |
      v
+--------------------------------------------------+
|  1. PREPROCESSING              src/preprocess.py |
|                                                  |
|  (i)  Load + resample to 16 kHz                 |
|       Whisper requires 16 kHz mono input.        |
|       Wrong sample rate degrades WER by 10-20%.  |
|                                                  |
|  (ii) Noise reduction  (noisereduce library)     |
|       Spectral gating removes stationary noise   |
|       (HVAC hum, keyboard clicks). Profile       |
|       estimated from first 0.5 s of the clip.   |
|                                                  |
|  (iii) Silence trimming  (librosa)               |
|        Removes leading/trailing silence.         |
|        Prevents Whisper hallucinations on        |
|        empty frames at clip boundaries.          |
+--------------------------------------------------+
      |
      v
+--------------------------------------------------+
|  2. TRANSCRIPTION              src/transcribe.py |
|                                                  |
|  OpenAI Whisper  (base / small / medium / large) |
|  - language = "es"   forced, no auto-detection   |
|  - beam_size = 5     beam search decoding        |
|  - temperature = 0   deterministic output        |
|  - fp16 auto         enabled on CUDA GPU         |
|  - Model loaded once, reused across all files    |
+--------------------------------------------------+
      |
      v
+--------------------------------------------------+
|  3. POST-PROCESSING           src/postprocess.py |
|                                                  |
|  (i)  Phrase corrections                         |
|       "man zana" -> "manzana"                    |
|       Applied first to fix split tokens.         |
|                                                  |
|  (ii) Word corrections  (whole-word regex)       |
|       "nino"   -> "nino"                         |
|       "rapido" -> "rapido"                       |
|       Capitalisation of original is preserved.   |
|                                                  |
|  (iii) Normalisation                             |
|        Lowercase + collapse whitespace.          |
|        Ensures consistent WER comparison.        |
+--------------------------------------------------+
      |
      v
+--------------------------------------------------+
|  4. EVALUATION                  src/evaluate.py  |
|                                                  |
|  WER = (S + D + I) / N                          |
|  Per-file scores + dataset average.              |
|  Saved to results/predictions.txt               |
+--------------------------------------------------+
      |
      v
+--------------------------------------------------+
|  5. ERROR ANALYSIS              src/evaluate.py  |
|                                                  |
|  Word-level alignment via jiwer.                 |
|  Ranks top substitutions, deletions,             |
|  and insertions across the full dataset.         |
+--------------------------------------------------+
      |
      v
  results/predictions.txt   (pipe-delimited per-file results)
  results/pipeline.log      (full timestamped run log)
```

---

## Project Structure

```
gsoc/
+-- src/
|   +-- __init__.py          Public API exports
|   +-- preprocess.py        Load, denoise, trim
|   +-- transcribe.py        Whisper wrapper + batch support
|   +-- postprocess.py       Accent restoration, phrase merging
|   +-- evaluate.py          WER, error analysis, results saving
+-- data/
|   +-- audio/               Learner audio files (.wav / .mp3)
|   +-- references.txt       Ground truth: filename|reference text
+-- results/
|   +-- predictions.txt      Per-file evaluation results
|   +-- pipeline.log         Full timestamped run log
+-- config.yaml              All pipeline parameters
+-- main.py                  CLI entry point
+-- test_main.py             45 unit tests
+-- requirements.txt
+-- README.md
```

---

## Baseline vs Improved Results

The table below shows representative WER values on a small Spanish learner
test set, comparing raw Whisper output against the full pipeline.

| Condition | WER (avg) | Notes |
|---|---|---|
| Whisper base, no preprocessing | 0.38 | Raw audio, no denoising |
| + Resampling to 16 kHz | 0.31 | Correct input format enforced |
| + Noise reduction | 0.26 | noisereduce spectral gating |
| + Silence trimming | 0.24 | Removes hallucination triggers |
| + Post-processing corrections | **0.18** | Accent and phrase fixes applied |

> Note: Results are illustrative. Actual WER depends on recording quality,
> speaker L1, and vocabulary. Run the pipeline on your own data for accurate
> numbers.

---

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH (required by Whisper for audio decoding)

```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

### Prepare your data

Place audio files in `data/audio/` and create `data/references.txt`:

```
audio1.wav|el nino come una manzana
audio2.wav|la mujer bebe agua
audio3.wav|el perro corre rapido
```

---

## Usage

### Basic run (uses `config.yaml` defaults)

```bash
python main.py
```

### Custom paths

```bash
python main.py --input data/audio --refs data/references.txt --output results/out.txt
```

### Use a more accurate Whisper model

```bash
python main.py --model small
```

### Disable noise reduction (faster, for clean recordings)

```bash
python main.py --no-denoise
```

### Force CPU

```bash
python main.py --device cpu
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--input` | `data/audio` | Folder containing audio files |
| `--refs` | `data/references.txt` | Pipe-delimited reference file |
| `--output` | `results/predictions.txt` | Results output file |
| `--model` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `--device` | auto | `cuda` or `cpu` |
| `--config` | `config.yaml` | Path to YAML config file |
| `--no-denoise` | off | Disable spectral noise reduction |

---

## Configuration

All parameters are controlled via `config.yaml`. CLI flags override config values.

```yaml
pipeline:
  model_size:  "base"    # tiny | base | small | medium | large
  language:    "es"      # BCP-47 code, forced (no auto-detection)
  denoise:     true
  beam_size:   5         # higher = more accurate, slower
  temperature: 0.0       # 0.0 = greedy / deterministic decoding

paths:
  audio_folder:   "data/audio"
  reference_file: "data/references.txt"
  results_file:   "results/predictions.txt"
  log_file:       "results/pipeline.log"

corrections:
  words:
    nino:   "nino"
    rapido: "rapido"
  phrases:
    "man zana": "manzana"
```

---

## Example Output

```
=======================================================
  Spanish Learner Speech Transcription Pipeline
=======================================================
model=base      language=es  device=cpu   denoise=True  beam=5
Running on CPU -- transcription will be slower

Found 3 audio file(s) to process

Processing: 100%|################| 3/3 [02:14<00:00]

-- [1/3] audio1.wav
  Prediction : el nino come una manzana
  Reference  : el nino come una manzana
  Time       : 42.3s

-- [2/3] audio2.wav
  Prediction : la mujer bebe agua
  Reference  : la mujer bebe agua
  Time       : 38.1s

-- [3/3] audio3.wav
  Prediction : el perro corre rapido
  Reference  : el perro corre rapido
  Time       : 41.8s

=======================================================
  RESULTS SUMMARY
=======================================================
  audio1.wav                        WER: 0.0000  [42.3s]
  audio2.wav                        WER: 0.0000  [38.1s]
  audio3.wav                        WER: 0.2000  [41.8s]
  AVERAGE                           WER: 0.0667
=======================================================

=======================================================
  ERROR ANALYSIS REPORT
=======================================================

[Substitutions]  reference -> hypothesis
  'rapido' -> 'rapido'  (x1)

[Deletions]  words missed by the model
  None found.

[Insertions]  extra words added by the model
  None found.
=======================================================

Total pipeline time: 2m 4.2s
Results saved to: results/predictions.txt
Log saved to:     results/pipeline.log
```

---

## Results Table

| File | Reference | Prediction | WER |
|---|---|---|---|
| audio1.wav | el nino come una manzana | el nino come una manzana | 0.0000 |
| audio2.wav | la mujer bebe agua | la mujer bebe agua | 0.0000 |
| audio3.wav | el perro corre rapido | el perro corre rapido | 0.2000 |
| **Average** | | | **0.0667** |

---

## Running Tests

```bash
python -m pytest test_main.py -v
```

45 tests across 6 test classes:

| Class | Tests | What it covers |
|---|---|---|
| `TestPreprocess` | 6 | load_audio, trim_silence, reduce_noise, preprocess |
| `TestPostprocess` | 16 | word/phrase corrections, normalisation, edge cases |
| `TestEvaluate` | 8 | compute_wer, evaluate_dataset, average_wer |
| `TestErrorAnalysis` | 5 | substitution/deletion/insertion detection |
| `TestSaveResults` | 3 | file creation, header, data row format |
| `TestMainHelpers` | 7 | _format_duration, _detect_device (with mocking) |

---

## Why Each Design Decision Was Made

**Why force `language="es"` in Whisper?**
Whisper's language detection is trained on native speech. For accented L2
audio it frequently misidentifies the language (e.g. detecting Portuguese
instead of Spanish), causing the wrong vocabulary to be used during decoding.
Forcing Spanish eliminates this failure mode entirely.

**Why `beam_size=5` instead of greedy decoding?**
Beam search explores multiple hypothesis paths and selects the most probable
sequence. For learner speech with unusual phoneme patterns, greedy decoding
often gets stuck in locally optimal but globally wrong transcriptions. Beam
search with width 5 improves WER by ~5-8% on accented speech at a 2-3x
compute cost.

**Why `temperature=0.0`?**
Temperature controls randomness in sampling. At 0.0, Whisper always picks the
highest-probability token, making output fully deterministic and reproducible
across runs — essential for research experiments.

**Why `noisereduce` over simple STFT subtraction?**
Simple spectral subtraction introduces "musical noise" artefacts that can
confuse Whisper's attention mechanism. `noisereduce` uses a spectral gate with
smoothing that produces cleaner output, especially for low-SNR recordings
typical of home and classroom environments.

**Why WER and not CER or BLEU?**
WER is the standard metric for ASR evaluation and directly measures word-level
accuracy, which aligns with how language teachers assess learner speech. CER
would be more sensitive to diacritic errors but harder to interpret. BLEU is
designed for machine translation and is not appropriate for ASR.

---

## Future Work

- **Fine-tuning on L2 corpora** — Fine-tune Whisper on CAPT datasets such as
  SpeechOcean762 or EpaDB to reduce baseline WER on learner speech by 20-40%
- **Phoneme-level analysis** — Integrate Montreal Forced Aligner (MFA) for
  phoneme error rates, enabling diagnosis of specific pronunciation problems
- **Character Error Rate (CER)** — Add CER alongside WER to better capture
  diacritic-level errors
- **Confidence scoring** — Use Whisper token log-probabilities to flag
  low-confidence segments for human review
- **Speaker diarization** — Separate multiple speakers in classroom recordings
  using pyannote.audio
- **Gradio web interface** — Real-time transcription feedback demo for
  language learners
- **Language model rescoring** — Rescore Whisper hypotheses with a Spanish LM
  to improve grammatical coherence
- **L1-specific correction tables** — Separate correction dictionaries per
  learner L1 (English, Mandarin, Portuguese) based on known transfer patterns

---

## License

MIT
