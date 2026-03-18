import wave, struct, math, os

sample_rate = 16000
frequency = 440.0
num_samples = sample_rate * 3

frames = struct.pack(
    '<' + 'h' * num_samples,
    *[int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate)) for i in range(num_samples)]
)

with wave.open('data/sample_audio.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(frames)

print('Created sample_audio.wav, size:', os.path.getsize('data/sample_audio.wav'), 'bytes')
