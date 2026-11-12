"""Decode any PyAV-readable audio to 16 kHz mono float32."""
SAMPLE_RATE = 16000


def decode(path):
    import av  # lazy: PyAV is only needed for real decoding
    import numpy as np

    frames = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            frames.extend(f.to_ndarray().flatten() for f in resampler.resample(frame))
        frames.extend(f.to_ndarray().flatten() for f in resampler.resample(None))
    if not frames:
        raise ValueError("Audio contained no decoded samples")
    return np.concatenate(frames).astype(np.float32)
