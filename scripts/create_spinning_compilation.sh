#!/bin/bash
# Create a video compilation of spinning gallery examples
# Phase mask on left, intensity on right, clean cuts between clips
# Single ffmpeg command - no intermediate files

set -e

OUTPUT_DIR="video_output"
mkdir -p "$OUTPUT_DIR"

# Spinning examples (path only)
SAMPLES=(
    "L1/spot_circular"
    "L1/linear_ramp_diagonal"
    "L1/ellipse_rotation"
    "L2/binary_grating_rotated"
    "L2/crossed_gratings"
    "L2/radial_sectors"
    "L2/spiral_grating"
    "L2/rings_and_sectors"
    "L3/grid_3x3_rotation"
    "L3/spots_brightness_gradient_x"
    "L3/spots_L_shape"
    "L3/spots_triangle_vertices"
    "L4/vortex_charge_1_rotation"
    "L4/vortex_charge_2"
    "L4/vortex_charge_3"
    "L4/bessel_order_1"
    "L4/lg_p1_l1"
    "L4/lg_p2_l2"
    "L4/vortex_plus_tilt"
)

FPS=30

# Build ffmpeg command with all inputs and complex filter
INPUTS=""
FILTER=""
n=${#SAMPLES[@]}

for i in "${!SAMPLES[@]}"; do
    path="${SAMPLES[$i]}"
    phase="static/assets/${path}_phase.gif"
    intensity="static/assets/${path}_intensity.gif"
    INPUTS="$INPUTS -i $phase -i $intensity"

    # Each pair: phase is input 2*i, intensity is input 2*i+1
    # setpts=PTS/3 speeds up 3x (original 10fps GIFs -> effectively 30fps playback)
    p=$((2*i))
    q=$((2*i+1))
    FILTER="$FILTER[$p:v]setpts=PTS/3,scale=256:256[p$i];[$q:v]setpts=PTS/3,scale=256:256[i$i];[p$i][i$i]hstack=inputs=2[clip$i];"
done

# Concatenate all clips
CONCAT_INPUTS=""
for i in "${!SAMPLES[@]}"; do
    CONCAT_INPUTS="${CONCAT_INPUTS}[clip$i]"
done
FILTER="${FILTER}${CONCAT_INPUTS}concat=n=$n:v=1:a=0[out]"

echo "Building video with $n clips at ${FPS}fps..."

ffmpeg -y $INPUTS \
    -filter_complex "$FILTER" \
    -map "[out]" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    -movflags +faststart \
    "$OUTPUT_DIR/spinning_compilation.mp4"

echo ""
echo "=== DONE ==="
echo "Video: $OUTPUT_DIR/spinning_compilation.mp4"
echo ""
ffprobe -v quiet -show_format "$OUTPUT_DIR/spinning_compilation.mp4" 2>/dev/null | grep -E "duration|bit_rate"

