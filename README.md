# Spanish Learner Speech Transcription Pipeline

A GSoC research prototype for automatic speech recognition (ASR) evaluation
of non-native Spanish learner audio using OpenAI Whisper.

---

## Problem Statement

Non-native speakers of Spanish exhibit systematic pronunciation errors that
challenge standard ASR systems:

- **Missing diacritics** — "nino" instead of "niño", "rapido" instead of "rápido"
- **Word fragmentation** — "man zana" instead of "manzana"
- **Accent interference** — phoneme substitutions driven by L1 transfer

This pipeline addresses these challenges through a modular preprocessing,
transcription, post-processing, and evaluation system designed to be
extensible for language learning research.

---

## Architecture

```
Audio File(s)
     │
     ▼
┌─────────────────────┐
│   Preprocessing     │  librosa — resample → denoise → trim silence
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Transcription     │  OpenAI Whisper (forced Spanish, beam search)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Post-processing    │  Phrase corrections → accent restoration → normalize
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│    Evaluation       │  Word Error Rate (WER) per file + dataset average
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│   Error Analysis    │  Substitutions / Deletions / Insertions summary
└─────────────────────┘
     │
     ▼
  results/predictions.txt
  results/pipeline.log
```

---

## Project Structure

```
gsoc/
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py    # Audio loading, denoising, silence trimming
│   ├── transcription.py    # Whisper model loading and batch transcription
│   ├── postprocessing.py   # Accent restoration and phrase correction
│   ├── evaluation.py       # WER computation and results saving
│   └── error_analysis.py   # Substitution/deletion/insertion analysis
├── data/
│   ├── references.txt      # Ground truth: filename|reference text
│   └── *.wav / *.mp3       # Learner audio files
├── results/
│   ├── predictions.txt     # Per-file results
│   └── pipeline.log        # Full run log
├── config.yaml             # All pipeline parameters
├── main.py                 # CLI entry point
├── test_main.py            # Unit tests (19 tests)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH
  ```bash
  # Windows (Chocolatey)
  choco install ffmpeg

  # macOS
  brew install ffmpeg

  # Ubuntu
  sudo apt install ffmpeg
  ```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Basic run (uses config.yaml defaults)

```bash
python main.py
```

### Custom input/output

```bash
python main.py --input data/audio --refs data/references.txt --output results/out.txt
```

### Use a larger Whisper model for better accuracy

```bash
python main.py --model small
```

### Disable noise reduction

```bash
python main.py --no-denoise
```

### All CLI options

```
--input     Path to audio folder          (default: data)
--refs      Path to references.txt        (default: data/references.txt)
--output    Path to results output file   (default: results/predictions.txt)
--model     Whisper model size            (default: base)
--config    Path to config YAML           (default: config.yaml)
--no-denoise  Disable spectral denoising
```

---

## Configuration

All parameters are configurable via `config.yaml`:

```yaml
pipeline:
  model_size: "base"     # tiny | base | small | medium | large
  language: "es"
  denoise: true
  beam_size: 5
  temperature: 0.0

corrections:
  words:
    nino: "niño"
    rapido: "rápido"
  phrases:
    "man zana": "manzana"
```

---

## References File Format

`data/references.txt` uses pipe-delimited format:

```
audio1.wav|el niño come una manzana
audio2.wav|la mujer bebe agua
audio3.wav|el perro corre rápido
```

---

## Example Output

```
2024-01-15 10:23:01 [INFO] Processing 3 audio file(s)...

── audio1.wav
  Prediction : el niño come una manzana
  Reference  : el niño come una manzana

── audio2.wav
  Prediction : la mujer bebe agua
  Reference  : la mujer bebe agua

=======================================================
  RESULTS SUMMARY
=======================================================
  audio1.wav                      WER: 0.0000
  audio2.wav                      WER: 0.0000
  audio3.wav                      WER: 0.2000
  AVERAGE                         WER: 0.0667
=======================================================
```

### Error Analysis Report

```
=======================================================
  ERROR ANALYSIS REPORT
=======================================================

[Substitutions] (reference → hypothesis)
  'rápido' → 'rapido'  (x1)

[Deletions] (words missed by model)
  None found.

[Insertions] (extra words added by model)
  None found.
=======================================================
```

---

## Results Table

| File        | Reference                    | Prediction                   | WER    |
|-------------|------------------------------|------------------------------|--------|
| audio1.wav  | el niño come una manzana     | el niño come una manzana     | 0.0000 |
| audio2.wav  | la mujer bebe agua           | la mujer bebe agua           | 0.0000 |
| audio3.wav  | el perro corre rápido        | el perro corre rapido        | 0.2000 |
| **Average** |                              |                              | **0.0667** |

---

## Running Tests

```bash
python -m pytest test_main.py -v
```

19 tests covering preprocessing, postprocessing, and evaluation modules.

---

## Future Work

- **Fine-tuning** — Fine-tune Whisper on L2 Spanish learner corpora (e.g. CAPT datasets)
- **Phoneme-level analysis** — Use forced alignment (Montreal Forced Aligner) for phoneme error rates
- **Confidence scoring** — Use Whisper token log-probabilities to flag uncertain segments
- **Speaker diarization** — Separate multiple speakers in classroom recordings
- **Web interface** — Gradio/Streamlit demo for real-time transcription feedback
- **BLEU / CER metrics** — Add Character Error Rate alongside WER
- **Language model rescoring** — Rescore Whisper hypotheses with a Spanish LM

---

## License

MIT
