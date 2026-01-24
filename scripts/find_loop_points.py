#!/usr/bin/env python3
"""Find loop points by sliding template and computing correlation.
Also searches for optimal start position.
Uses librosa to load MP3 directly at native sample rate for accurate timestamps."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import subprocess

# Load audio directly from MP3 using ffmpeg to ensure timestamps match
MP3_FILE = "video_output/Dead Or Alive - You Spin Me Round (Like a Record) (Official Video) [PGNiXGX2nLU].mp3"
SAMPLE_RATE = 44100  # Native MP3 sample rate
DURATION = 25  # seconds to analyze

# Extract audio using ffmpeg to raw PCM (mono, float32)
RAW_FILE = "video_output/audio_25s_44100.raw"
cmd = f'ffmpeg -y -i "{MP3_FILE}" -t {DURATION} -ac 1 -ar {SAMPLE_RATE} -f f32le "{RAW_FILE}"'
print(f"Extracting audio: {cmd}")
subprocess.run(cmd, shell=True, capture_output=True)

audio = np.fromfile(RAW_FILE, dtype=np.float32)
print(f"Audio length: {len(audio)} samples = {len(audio)/SAMPLE_RATE:.2f}s")

# Parameters
template_duration = 0.1  # seconds - small for fine alignment
template_samples = int(template_duration * SAMPLE_RATE)
target_video_duration = 12.83

# Search for best start position and loop end around the two candidate regions
# Region 1: loop end ~8.95s, Region 2: loop end ~9.4s
start_search_range = np.arange(1.1, 1.35, 0.01)  # vary start from 1.1s to 1.35s
end_search_regions = [(8.7, 9.2), (9.2, 9.7)]  # two candidate end regions

print(f"\nSearching for optimal loop with template duration {template_duration}s")
print(f"Start positions: {start_search_range[0]:.2f}s to {start_search_range[-1]:.2f}s")

best_results = []

for region_idx, (end_min, end_max) in enumerate(end_search_regions):
    print(f"\n=== Region {region_idx+1}: loop end {end_min}s - {end_max}s ===")
    region_best = {'corr': 0}

    for template_start in start_search_range:
        template_start_sample = int(template_start * SAMPLE_RATE)
        template = audio[template_start_sample:template_start_sample + template_samples]

        # Compute correlation
        correlation = np.correlate(audio, template, mode='valid')
        template_energy = np.sum(template**2)
        running_energy = np.convolve(audio**2, np.ones(template_samples), mode='valid')
        running_energy = np.maximum(running_energy, 1e-10)
        correlation = correlation / np.sqrt(template_energy * running_energy)

        # Search in the target end region
        start_idx = int((end_min - template_start) * SAMPLE_RATE)
        end_idx = int((end_max - template_start) * SAMPLE_RATE)
        start_idx = max(0, start_idx)
        end_idx = min(len(correlation), end_idx)

        if start_idx >= end_idx:
            continue

        region_corr = correlation[start_idx:end_idx]
        peaks, _ = find_peaks(region_corr, height=0.2, distance=int(0.02 * SAMPLE_RATE))

        for peak in peaks:
            actual_idx = start_idx + peak
            loop_end = template_start + actual_idx / SAMPLE_RATE
            corr_val = region_corr[peak]
            loop_duration = loop_end - template_start
            loop_2x = loop_duration * 2
            speedup = loop_2x / target_video_duration

            if corr_val > region_best['corr']:
                region_best = {
                    'start': template_start,
                    'end': loop_end,
                    'duration': loop_duration,
                    'loop_2x': loop_2x,
                    'speedup': speedup,
                    'corr': corr_val
                }

    if region_best['corr'] > 0:
        best_results.append(region_best)
        r = region_best
        print(f"  Best: {r['start']:.4f}s -> {r['end']:.4f}s")
        print(f"        duration={r['duration']:.4f}s, 2x={r['loop_2x']:.4f}s, speedup={r['speedup']:.4f}x, corr={r['corr']:.4f}")

print("\n=== BEST LOOP CANDIDATES ===")
for i, r in enumerate(best_results):
    print(f"Option {i+1}: {r['start']:.4f}s -> {r['end']:.4f}s, duration={r['duration']:.4f}s, 2x={r['loop_2x']:.4f}s, speedup={r['speedup']:.4f}x, corr={r['corr']:.4f}")

print("\n=== FFMPEG COMMANDS (using -ss after -i for sample-accurate seeking) ===")
for i, r in enumerate(best_results):
    start = r['start']
    duration = r['duration']
    speedup = r['speedup']
    outfile = f"video_output/loop_opt{i+1}_{int(start*1000)}_{int(r['end']*1000)}_2x.mp3"
    # Use -ss after -i for accurate seeking, and use atrim filter for precise cutting
    cmd = f'ffmpeg -y -i "video_output/Dead Or Alive - You Spin Me Round (Like a Record) (Official Video) [PGNiXGX2nLU].mp3" -af "atrim=start={start:.6f}:duration={duration:.6f},asetpts=PTS-STARTPTS,aloop=loop=1:size={int(duration*44100)},atempo={speedup:.6f}" -t 13 -b:a 192k "{outfile}"'
    print(f"\n# Option {i+1}: start={start:.6f}s, duration={duration:.6f}s, speedup={speedup:.6f}x")
    print(cmd)

# Compute normalized cross-correlation by sliding template
# We want to find where in the audio the template repeats
correlation = np.correlate(audio, template, mode='valid')

# Normalize - use abs() of template for envelope matching
template_energy = np.sum(template**2)
# Use running energy for normalization
running_energy = np.convolve(audio**2, np.ones(template_samples), mode='valid')
# Avoid division by zero
running_energy = np.maximum(running_energy, 1e-10)
correlation = correlation / np.sqrt(template_energy * running_energy)

# Time axis
time = np.arange(len(correlation)) / SAMPLE_RATE

# Find peaks (potential loop points)
# Skip first 1 second to avoid self-match
min_offset = int(1.0 * SAMPLE_RATE)
correlation_search = correlation[min_offset:]
time_search = time[min_offset:]

# Find top peaks - use smaller distance for finer resolution
from scipy.signal import find_peaks
peaks, properties = find_peaks(correlation_search, height=0.2, distance=int(0.05 * SAMPLE_RATE))

print(f"\n=== Potential loop points (high correlation with template at {template_start}s) ===")
for i, peak in enumerate(peaks[:20]):  # Top 20
    t = time_search[peak] + template_start  # Add template offset to get actual time
    corr = correlation_search[peak]
    print(f"  {t:.3f}s  (correlation: {corr:.3f})")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Waveform
ax1 = axes[0]
t_audio = np.arange(len(audio)) / SAMPLE_RATE
ax1.plot(t_audio, audio, 'b-', linewidth=0.3)
ax1.axvspan(template_start, template_start + template_duration, alpha=0.3, color='red', label=f'Template ({template_start}s - {template_start+template_duration}s)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Audio waveform (first 20s)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Correlation
ax2 = axes[1]
ax2.plot(time + template_start, correlation, 'g-', linewidth=0.5)
ax2.axhline(y=0.3, color='r', linestyle='--', alpha=0.5, label='Threshold 0.3')
for peak in peaks[:20]:
    t = time_search[peak] + template_start
    ax2.axvline(x=t, color='orange', alpha=0.5, linewidth=1)
ax2.set_xlabel('Time offset (s)')
ax2.set_ylabel('Normalized correlation')
ax2.set_title('Cross-correlation (sliding template)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('video_output/loop_analysis.png', dpi=150)
print("\nPlot saved: video_output/loop_analysis.png")

# Suggest best loop for ~12.8s video
print("\n=== Suggested loops for 12.83s video ===")
target = 12.83

# Fine search around the two candidate loop points: 8.956s and 9.424s
print("\n--- Fine search around 8.956s (±0.2s) ---")
for peak in peaks:
    t = time_search[peak] + template_start  # Actual loop end point
    loop_duration = t - template_start
    loop_2x = loop_duration * 2
    speedup = loop_2x / target
    if 8.7 < t < 9.2:
        print(f"  Loop {template_start:.1f}s -> {t:.4f}s, duration {loop_duration:.3f}s, 2x={loop_2x:.3f}s, speedup {speedup:.4f}x (corr: {correlation_search[peak]:.3f})")

print("\n--- Fine search around 9.424s (±0.2s) ---")
for peak in peaks:
    t = time_search[peak] + template_start  # Actual loop end point
    loop_duration = t - template_start
    loop_2x = loop_duration * 2
    speedup = loop_2x / target
    if 9.2 < t < 9.7:
        print(f"  Loop {template_start:.1f}s -> {t:.4f}s, duration {loop_duration:.3f}s, 2x={loop_2x:.3f}s, speedup {speedup:.4f}x (corr: {correlation_search[peak]:.3f})")

# Option 2: Original longer loops (~14-18s range)
print("\n--- Longer loops (14-18s, ~1.1-1.4x speedup) ---")
for peak in peaks:
    t = time_search[peak] + template_start  # Actual loop end point
    if 14 < t < 18:
        speed = t / target
        print(f"  Loop 0 -> {t:.3f}s, speed {speed:.3f}x to fit 12.83s (corr: {correlation_search[peak]:.3f})")

