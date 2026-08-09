"""Build a 30-second crescendo from real, attribution-free animal recordings.

The source files are from Wikimedia Commons and are either CC0 or public
domain. Run from the repository root; the output is written beside the source
folder as ``media/audio/Animal_Sounds_Crescendo.mp3``.
"""
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np


SR = 48_000
DURATION = 30.0
HERE = Path(__file__).parent
SAMPLE_DIR = HERE / "media" / "audio" / "animal_samples"
OUT = HERE / "media" / "audio" / "Animal_Sounds_Crescendo.mp3"

# License status and source URLs are retained here for the production record.
# CC0 and public-domain works require no attribution in the YouTube description.
SOURCES = {
    "nightjar.ogg": ("CC0", "https://commons.wikimedia.org/wiki/File:Caprimulgus_europaeus.ogg"),
    "cicadas.ogg": ("Public domain", "https://commons.wikimedia.org/wiki/File:Cicadas_in_Greece.ogg"),
    "donkey.wav": ("CC0", "https://commons.wikimedia.org/wiki/File:157763_felix-blume_a-donkey-is-braying-in-his-enclosure-in-south-of-france.wav"),
    "goats.ogg": ("Public domain", "https://commons.wikimedia.org/wiki/File:Herd_of_goats_bleating.ogg"),
    "wolves.ogg": ("Public domain", "https://commons.wikimedia.org/wiki/File:Wolf_howls.ogg"),
    "elephant.ogg": ("CC0", "https://commons.wikimedia.org/wiki/File:Elephant_voice_-_trumpeting.ogg"),
}


def decode(name):
    raw = subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(SAMPLE_DIR / name),
        "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"
    ], check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def excerpt(x, seconds, search_from=0.0):
    """Select the highest-energy window after ``search_from``."""
    n = min(len(x), int(seconds * SR))
    start = min(len(x) - n, int(search_from * SR))
    region = x[start:]
    hop = SR // 5
    if len(region) > n:
        candidates = np.arange(0, len(region) - n + 1, hop)
        cumulative = np.concatenate(([0.0], np.cumsum(region * region)))
        scores = cumulative[candidates + n] - cumulative[candidates]
        start += int(candidates[int(np.argmax(scores))])
    y = x[start:start + n].copy()
    y -= y.mean()
    rms = np.sqrt(np.mean(y * y)) + 1e-9
    y /= rms
    fade = min(len(y) // 4, int(.12 * SR))
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y


def add(mix, sound, at, gain, pan=0.0):
    i = int(at * SR)
    sound = sound[:len(mix) - i]
    if len(sound) <= 0:
        return
    left, right = np.sqrt((1 - pan) / 2), np.sqrt((1 + pan) / 2)
    mix[i:i + len(sound), 0] += sound * gain * left
    mix[i:i + len(sound), 1] += sound * gain * right


def rattlesnake(seconds=3.6):
    """Original synthetic rattle: irregular bursts of dry, band-passed noise."""
    rng = np.random.default_rng(815)
    n = int(seconds * SR)
    out = np.zeros(n)
    for at in np.arange(0, seconds, .038):
        width = int(rng.uniform(.010, .025) * SR)
        i = int((at + rng.uniform(-.006, .006)) * SR)
        noise = rng.standard_normal(width)
        spectrum = np.fft.rfft(noise)
        freq = np.fft.rfftfreq(width, 1 / SR)
        band = np.exp(-((freq - rng.uniform(2900, 4300)) / 1900) ** 2)
        grain = np.fft.irfft(spectrum * band, width)
        grain *= np.sin(np.linspace(0, np.pi, width)) ** .6
        if i >= 0:
            out[i:min(n, i + width)] += grain[:max(0, min(width, n - i))]
    out /= np.sqrt(np.mean(out * out)) + 1e-9
    out *= np.sin(np.linspace(0, np.pi, n)) ** .35
    return out


def build():
    audio = {name: decode(name) for name in SOURCES}
    calls = {
        "nightjar": excerpt(audio["nightjar.ogg"], 2.8),
        "cicadas": excerpt(audio["cicadas.ogg"], 8.0),
        "goats": excerpt(audio["goats.ogg"], 2.6),
        "donkey": excerpt(audio["donkey.wav"], 3.3),
        "wolves": excerpt(audio["wolves.ogg"], 4.8),
        "elephant": excerpt(audio["elephant.ogg"], 1.4),
        "rattlesnake": rattlesnake(),
    }
    mix = np.zeros((int(DURATION * SR), 2), dtype=np.float64)

    # Start with one distant bird. Add species, overlap and level toward a
    # dense final chorus, ending with the elephant trumpet.
    events = [
        ("nightjar", 0.4, .019, -.55), ("nightjar", 3.0, .021, .45),
        ("cicadas", 5.5, .012, -.10), ("nightjar", 6.8, .027, .65),
        ("nightjar", 8.6, .028, -.40), ("cicadas", 9.5, .016, .10),
        ("goats", 10.5, .032, .30), ("cicadas", 11.2, .017, .10),
        ("nightjar", 13.0, .038, .55), ("donkey", 14.2, .040, -.30),
        ("goats", 16.3, .044, .55), ("rattlesnake", 17.1, .026, -.60),
        ("nightjar", 18.2, .042, .20),
        ("wolves", 19.0, .045, -.25),
        ("donkey", 20.8, .047, .58), ("nightjar", 21.8, .045, -.70),
        ("goats", 22.2, .048, .30), ("rattlesnake", 22.7, .037, -.45),
        ("wolves", 23.2, .055, .38), ("cicadas", 21.9, .024, .05),
        ("nightjar", 23.8, .049, -.30),
        ("donkey", 24.7, .053, -.58), ("goats", 25.2, .054, .65),
        ("rattlesnake", 25.6, .045, .18), ("nightjar", 26.1, .050, -.75),
        ("wolves", 25.1, .060, -.15), ("elephant", 28.35, .088, .05),
    ]
    for name, at, gain, pan in events:
        add(mix, calls[name], at, gain, pan)

    # Continuous gain growth makes the structure read unmistakably as a
    # crescendo even though individual source recordings vary in dynamics.
    ramp = np.linspace(.28, 1.0, len(mix)) ** 1.25
    mix *= ramp[:, None]
    mix *= .82 / (np.max(np.abs(mix)) + 1e-9)
    return mix


def main():
    missing = [name for name in SOURCES if not (SAMPLE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing animal samples: {', '.join(missing)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.int16(np.clip(build(), -1, 1) * 32767)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "animal_crescendo.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(SR)
            wav.writeframes(pcm.tobytes())
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-b:a", "256k", str(OUT)
        ], check=True)
    print(OUT)


if __name__ == "__main__":
    main()
