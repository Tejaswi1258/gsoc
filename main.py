import whisper
import librosa
from jiwer import wer
import os
import shutil

# -----------------------------
# AUDIO PREPROCESSING
# -----------------------------
def preprocess_audio(file_path):
    """
    Load audio and remove silence
    """
    audio, sr = librosa.load(file_path, sr=16000)

    # Trim silence from beginning and end
    audio_trimmed, _ = librosa.effects.trim(audio)

    return audio_trimmed, sr


# -----------------------------
# TRANSCRIPTION
# -----------------------------
def transcribe_audio(file_path):
    """
    Convert speech to text
    """

    # kept for backwards-compatibility; prefer passing a loaded model
    model = whisper.load_model("base")

    result = model.transcribe(file_path, language="es")

    return result["text"]


# -----------------------------
# POST PROCESSING
# -----------------------------
def postprocess_text(text):
    """
    Correct common learner transcription errors
    """

    corrections = {
        "nino": "niño",
        "man zana": "manzana",
        "rapido": "rápido",
        "senor": "señor"
    }

    words = text.split()

    corrected_text = " ".join(corrections.get(word, word) for word in words)

    return corrected_text


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(reference, prediction):
    """
    Calculate Word Error Rate
    """

    score = wer(reference, prediction)

    return score


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():

    # point to the audio directory
    audio_folder = "data"
    reference_file = "data/references.txt"

    # Load references
    references = {}

    try:
        with open(reference_file) as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split("|", 1)
                if len(parts) != 2:
                    continue
                audio, text = parts
                references[audio] = text
    except FileNotFoundError:
        print(f"Reference file not found: {reference_file}")
        return

    total_wer = 0
    count = 0

    predictions = []

    print("Starting transcription pipeline...\n")

    # verify ffmpeg is available (used by whisper for audio decoding)
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH. Install ffmpeg and ensure it's available in PATH.")
        print("On Windows, you can install via Chocolatey: 'choco install ffmpeg' or download from ffmpeg.org.")
        return

    # load model once to avoid reloading for each file
    try:
        model = whisper.load_model("base")
    except Exception as e:
        print("Failed to load Whisper model:", e)
        return

    for audio_file in os.listdir(audio_folder):
        # skip non-audio files and directories
        if not audio_file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            continue

        audio_path = os.path.join(audio_folder, audio_file)

        if not os.path.isfile(audio_path):
            continue

        print("Processing:", audio_file)

        # skip empty or zero-length files
        try:
            if os.path.getsize(audio_path) == 0:
                print(f"Skipping empty audio file: {audio_file}")
                continue
        except OSError:
            print(f"Could not access file size for: {audio_file}")
            continue

        # transcription
        try:
            prediction = model.transcribe(audio_path, language="es")["text"]
        except Exception as e:
            print(f"Transcription failed for {audio_file}: {e}")
            continue

        # postprocess
        prediction = postprocess_text(prediction)

        reference = references.get(audio_file, "")

        error = evaluate(reference, prediction)

        total_wer += error
        count += 1

        predictions.append(f"{audio_file}|{prediction}|WER:{error}")

        print("Prediction:", prediction)
        print("Reference:", reference)
        print("WER:", error)
        print()

    if count == 0:
        print("No audio files processed.")
        return

    avg_wer = total_wer / count

    print("Average WER:", avg_wer)

    # save results
    os.makedirs("results", exist_ok=True)

    with open("results/predictions.txt", "w") as f:
        for line in predictions:
            f.write(line + "\n")


if __name__ == "__main__":
    main()