#!/usr/bin/env python3
"""Create a seamless audio loop with crossfade at the loop point.
Uses numpy for sample-accurate audio processing."""

import numpy as np
import subprocess
import struct

# Constants
MP3_FILE = "video_output/Dead Or Alive - You Spin Me Round (Like a Record) (Official Video) [PGNiXGX2nLU].mp3"
SAMPLE_RATE = 44100
MP3_START_OFFSET = 0.025057

# Beat-aligned loop points (from find_loop_beats.py)
# Option 1: 4 measures, 1.1146s -> 8.6030s, speedup 1.167x
LOOP_START = 1.1146
LOOP_END = 8.6030
TARGET_DURATION = 12.83  # video duration

# Crossfade duration
CROSSFADE_SEC = 0.05  # 50ms crossfade

# Extract audio segment with extra at the end for crossfade
extract_start = LOOP_START + MP3_START_OFFSET
loop_duration = LOOP_END - LOOP_START
extract_duration = loop_duration + CROSSFADE_SEC

RAW_FILE = "video_output/loop_segment.raw"
cmd = f'ffmpeg -y -i "{MP3_FILE}" -ss {extract_start:.6f} -t {extract_duration:.6f} -ac 1 -ar {SAMPLE_RATE} -f f32le "{RAW_FILE}"'
print(f"Extracting segment: {extract_start:.4f}s, duration: {extract_duration:.4f}s")
subprocess.run(cmd, shell=True, capture_output=True)

audio = np.fromfile(RAW_FILE, dtype=np.float32)
print(f"Loaded {len(audio)} samples = {len(audio)/SAMPLE_RATE:.4f}s")

# Create loopable segment with crossfade
# The idea: 
# - Take the main segment [0 : loop_duration]
# - Create a crossfade region at the end where we blend:
#   - End of segment (fading out)
#   - Beginning of segment (fading in)

loop_samples = int(loop_duration * SAMPLE_RATE)
cf_samples = int(CROSSFADE_SEC * SAMPLE_RATE)

print(f"Loop samples: {loop_samples}, crossfade samples: {cf_samples}")

# Main segment (will be the repeating unit)
segment = audio[:loop_samples].copy()

# Create crossfade region
# Fade out the end, fade in the beginning
fade_out = np.linspace(1, 0, cf_samples)  # linear fade
fade_in = np.linspace(0, 1, cf_samples)

# Apply crossfade at the end of segment
# segment[-cf_samples:] will fade out
# audio[loop_samples : loop_samples+cf_samples] will fade in
# Actually, for a seamless loop, we blend:
# - End of segment: segment[-cf_samples:] fading out
# - Beginning of segment: segment[:cf_samples] fading in

# First, create the looped audio (2 repetitions)
loop_2x = np.zeros(loop_samples * 2, dtype=np.float32)
loop_2x[:loop_samples] = segment
loop_2x[loop_samples:] = segment

# Apply crossfade at the transition point (at loop_samples)
# Fade out the end of first repetition
# Fade in the start of second repetition
transition_start = loop_samples - cf_samples
transition_end = loop_samples + cf_samples

# Create blended transition region
for i in range(cf_samples):
    # Position in the fade
    t = i / cf_samples
    # Blending: end of first loop fading out, start of second loop fading in
    idx = loop_samples - cf_samples + i
    orig_end = loop_2x[idx]  # end of first segment
    orig_start = segment[i]   # start of segment (second rep)
    # Crossfade
    loop_2x[idx] = orig_end * (1 - t) + orig_start * t

# Shift the second part to overlap properly
# Actually, we need to shorten the result because of the overlap
loop_seamless = np.zeros(loop_samples * 2 - cf_samples, dtype=np.float32)
loop_seamless[:loop_samples - cf_samples] = segment[:-cf_samples]

# Crossfade region
for i in range(cf_samples):
    t = i / cf_samples
    loop_seamless[loop_samples - cf_samples + i] = segment[-cf_samples + i] * (1 - t) + segment[i] * t

# Second segment (minus the crossfaded beginning)
loop_seamless[loop_samples:] = segment[cf_samples:]

print(f"Seamless loop length: {len(loop_seamless)} samples = {len(loop_seamless)/SAMPLE_RATE:.4f}s")

# Calculate required speedup
loop_2x_duration = len(loop_seamless) / SAMPLE_RATE
speedup = loop_2x_duration / TARGET_DURATION
print(f"2x loop duration: {loop_2x_duration:.4f}s, speedup for {TARGET_DURATION}s video: {speedup:.4f}x")

# Save as raw audio
RAW_OUT = "video_output/loop_seamless_2x.raw"
loop_seamless.tofile(RAW_OUT)

# Convert to MP3 with tempo adjustment
OUT_MP3 = "video_output/loop_seamless_4meas_cf.mp3"
cmd = f'ffmpeg -y -f f32le -ar {SAMPLE_RATE} -ac 1 -i "{RAW_OUT}" -af "atempo={speedup:.6f}" -t 13.5 -b:a 192k "{OUT_MP3}"'
print(f"\nCreating MP3 with tempo adjustment...")
print(cmd)
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error: {result.stderr}")
else:
    print(f"Created: {OUT_MP3}")

# Also create a version without tempo adjustment for inspection
OUT_MP3_ORIG = "video_output/loop_seamless_4meas_cf_orig_tempo.mp3"
cmd = f'ffmpeg -y -f f32le -ar {SAMPLE_RATE} -ac 1 -i "{RAW_OUT}" -b:a 192k "{OUT_MP3_ORIG}"'
subprocess.run(cmd, shell=True, capture_output=True)
print(f"Created (original tempo): {OUT_MP3_ORIG}")

print("\n=== SUMMARY ===")
print(f"Loop segment: {LOOP_START:.4f}s -> {LOOP_END:.4f}s ({loop_duration:.4f}s)")
print(f"Crossfade: {CROSSFADE_SEC*1000:.0f}ms")
print(f"2x seamless loop: {loop_2x_duration:.4f}s")
print(f"Speedup for {TARGET_DURATION}s video: {speedup:.4f}x")

