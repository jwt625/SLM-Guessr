#!/usr/bin/env python3
"""Generate interactive HTML waveform viewer."""

import numpy as np
import subprocess
import json

# Constants
MP3_FILE = "video_output/Dead Or Alive - You Spin Me Round (Like a Record) (Official Video) [PGNiXGX2nLU].mp3"
SAMPLE_RATE = 44100
DURATION = 30

# Previously found loop points
LOOP_CANDIDATES = [
    {'start': 1.1146, 'end': 9.0674, 'label': '17 beats (1.24x)', 'color': '#e41a1c'},
    {'start': 1.1400, 'end': 8.8356, 'label': 'corr opt1 (1.20x)', 'color': '#ff7f00'},
    {'start': 1.1000, 'end': 9.6924, 'label': 'corr opt2 (1.34x)', 'color': '#4daf4a'},
    {'start': 1.1146, 'end': 8.6030, 'label': '4 meas (1.17x)', 'color': '#984ea3'},
]

# Extract audio - MONO mixdown, 44100 Hz, 32-bit float
RAW_FILE = "video_output/audio_beats.raw"
cmd = f'ffmpeg -y -i "{MP3_FILE}" -t {DURATION} -ac 1 -ar {SAMPLE_RATE} -f f32le "{RAW_FILE}"'
print("Extracting audio (mono mixdown, 44100 Hz)...")
subprocess.run(cmd, shell=True, capture_output=True)

audio = np.fromfile(RAW_FILE, dtype=np.float32)
print(f"Audio: {len(audio)} samples = {len(audio)/SAMPLE_RATE:.2f}s")
print(f"Sample rate: {SAMPLE_RATE} Hz, time per sample: {1000/SAMPLE_RATE:.4f} ms")

# Full resolution
time_full = (np.arange(len(audio)) / SAMPLE_RATE).tolist()
audio_full = audio.tolist()

# Convert MP3 to base64 for embedding
import base64
with open(MP3_FILE, 'rb') as f:
    mp3_b64 = base64.b64encode(f.read()).decode('ascii')
print(f"Embedded MP3: {len(mp3_b64)//1024} KB")

# Generate HTML
html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Waveform Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #plot {{ width: 100%; height: 500px; }}
        .info {{ margin: 10px 0; padding: 10px; background: #f0f0f0; }}
        .legend {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 10px 0; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-color {{ width: 20px; height: 3px; }}
        .audio-controls {{ margin: 20px 0; padding: 15px; background: #e8e8e8; border-radius: 5px; }}
        .audio-controls audio {{ width: 100%; }}
        .time-display {{ font-size: 24px; font-family: monospace; margin: 10px 0; }}
        .click-info {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>Waveform Viewer - You Spin Me Round</h1>
    <div class="audio-controls">
        <audio id="audio" controls>
            <source src="data:audio/mpeg;base64,{mp3_b64}" type="audio/mpeg">
        </audio>
        <div class="time-display">Current: <span id="currentTime">0.000</span>s</div>
        <div class="click-info">Click on waveform to seek. Playhead shown as black vertical line.</div>
    </div>
    <div class="info">
        <b>Instructions:</b> Drag to zoom, double-click to reset. Scroll to zoom. Hover for exact time.
    </div>
    <div class="legend">
        <b>Loop candidates:</b>
        {"".join(f'<span class="legend-item"><span class="legend-color" style="background:{lc["color"]}"></span>{lc["label"]}: {lc["start"]:.3f}s → {lc["end"]:.3f}s</span>' for lc in LOOP_CANDIDATES)}
    </div>
    <div id="plot"></div>
    <script>
        const time = {json.dumps(time_full)};
        const audio = {json.dumps(audio_full)};
        const candidates = {json.dumps(LOOP_CANDIDATES)};

        const trace = {{
            x: time,
            y: audio,
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#1f77b4', width: 0.5 }},
            name: 'Waveform'
        }};

        const shapes = [];
        const annotations = [];
        candidates.forEach(c => {{
            // Start line (dashed)
            shapes.push({{
                type: 'line',
                x0: c.start, x1: c.start,
                y0: -1, y1: 1,
                line: {{ color: c.color, width: 2, dash: 'dash' }}
            }});
            // End line (solid)
            shapes.push({{
                type: 'line',
                x0: c.end, x1: c.end,
                y0: -1, y1: 1,
                line: {{ color: c.color, width: 2 }}
            }});
            // Labels
            annotations.push({{
                x: c.start, y: 0.9,
                text: 'S:' + c.label.split(' ')[0],
                showarrow: false,
                font: {{ size: 9, color: c.color }}
            }});
            annotations.push({{
                x: c.end, y: -0.9,
                text: 'E:' + c.label.split(' ')[0],
                showarrow: false,
                font: {{ size: 9, color: c.color }}
            }});
        }});

        const layout = {{
            title: 'Audio Waveform - Mono Mixdown, 44100 Hz ({len(audio)} samples, {1000/SAMPLE_RATE:.4f} ms/sample)',
            xaxis: {{ title: 'Time (seconds)', rangeslider: {{}} }},
            yaxis: {{ title: 'Amplitude (normalized float32, -1 to +1)', range: [-1, 1] }},
            shapes: shapes,
            annotations: annotations,
            hovermode: 'x unified'
        }};

        Plotly.newPlot('plot', [trace], layout, {{responsive: true}});

        // Audio sync using CSS overlay (fast, no redraw)
        const audioEl = document.getElementById('audio');
        const timeDisplay = document.getElementById('currentTime');
        const plotDiv = document.getElementById('plot');

        // Create playhead overlay
        const playhead = document.createElement('div');
        playhead.style.cssText = 'position:absolute;width:2px;background:red;pointer-events:none;z-index:1000;';
        plotDiv.style.position = 'relative';
        plotDiv.appendChild(playhead);

        function updatePlayhead() {{
            const currentTime = audioEl.currentTime;
            timeDisplay.textContent = currentTime.toFixed(3);

            // Get Plotly's internal axis object and layout
            const layout = plotDiv._fullLayout;
            if (!layout || !layout.xaxis) return;

            const xa = layout.xaxis;
            const xRange = xa.range;

            // Use Plotly's coordinate conversion: data coords -> pixel coords
            // xa.l2p converts from data value to pixels relative to plot area left edge
            // xa._offset is the left edge of the plot area
            const xPx = xa.l2p(currentTime) + xa._offset;

            // Get plot area height from yaxis
            const ya = layout.yaxis;
            const plotTop = ya._offset;
            const plotHeight = ya._length;

            // Position playhead
            playhead.style.left = xPx + 'px';
            playhead.style.top = plotTop + 'px';
            playhead.style.height = plotHeight + 'px';

            // Hide if outside visible range
            const xFrac = (currentTime - xRange[0]) / (xRange[1] - xRange[0]);
            playhead.style.display = (xFrac >= 0 && xFrac <= 1) ? 'block' : 'none';
        }}

        // Use requestAnimationFrame for smooth updates
        function animatePlayhead() {{
            updatePlayhead();
            requestAnimationFrame(animatePlayhead);
        }}
        animatePlayhead();

        // Click on plot to seek
        plotDiv.on('plotly_click', (data) => {{
            if (data.points && data.points.length > 0) {{
                audioEl.currentTime = data.points[0].x;
            }}
        }});

        // Update on zoom/pan
        plotDiv.on('plotly_relayout', updatePlayhead);
    </script>
</body>
</html>
'''

out_file = "video_output/waveform_viewer.html"
with open(out_file, 'w') as f:
    f.write(html)

print(f"Created: {out_file}")
