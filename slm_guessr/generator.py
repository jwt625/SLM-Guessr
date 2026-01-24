"""
Sample Generator for SLM-Guessr

Generates animated GIFs and samples.json manifest for the training gallery.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Callable
import sys

# Add parent dir for gs_algorithms import
sys.path.insert(0, str(Path(__file__).parent.parent))
from gs_algorithms import standard_gs

from .patterns import (
    create_gaussian_input,
    create_uniform_phase,
    create_linear_ramp,
    create_quadratic_phase,
    create_cubic_phase,
    create_spot_target,
    create_gaussian_spot_target,
    create_rectangular_slab_target,
    create_line_target,
    create_band_target,
    create_ellipse_target,
    create_ring_target,
    create_ring_aperture_phase,
    create_cylindrical_lens_phase,
    create_two_spots_target,
    create_binary_grating,
    create_sinusoidal_grating,
    create_blazed_grating,
    create_checkerboard,
    create_crossed_gratings,
    create_multi_frequency_grating,
    compute_intensity,
)

from .patterns_L2 import (
    create_concentric_rings_binary,
    create_concentric_rings_sinusoidal,
    create_radial_sectors,
    create_spiral_grating,
    create_hexagonal_lattice,
    create_triangular_lattice,
    create_grating_with_defect,
    create_chirped_grating,
    create_duty_cycle_grating,
    create_amplitude_modulated_grating,
    create_blazed_grating_variable,
    create_rings_and_sectors,
)

from .patterns_L3 import (
    create_grid_spots,
    create_spots_at_positions,
    create_random_spot_positions,
)

from .patterns_L4 import (
    create_vortex_phase,
    create_axicon_phase,
    create_laguerre_gaussian_phase,
    create_laguerre_gaussian_target,
    create_bessel_target,
)

# Use PIL for GIF generation
from PIL import Image, ImageDraw, ImageFont


GRID_SIZE = 256
GS_ITERATIONS = 50
GIF_DURATION_MS = 100  # ms per frame


@dataclass
class SampleConfig:
    """Configuration for a training sample."""
    id: str
    level: int
    category: str
    name: str
    description: str
    generator: Callable  # Function that returns list of (phase, intensity) frames
    parameters: dict = None
    duration_ms: int = None  # Override GIF frame duration (default: GIF_DURATION_MS)
    intensity_zoom: float = None  # Zoom factor for intensity images (e.g., 5.0 for 5x zoom)


def normalize_for_image(arr: np.ndarray, is_phase: bool = False) -> np.ndarray:
    """Normalize array to 0-255 uint8 for image saving."""
    if is_phase:
        # Phase: map [-pi, pi] to [0, 255]
        normalized = (arr + np.pi) / (2 * np.pi)
    else:
        # Intensity: normalize to [0, 1] then scale
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            normalized = np.zeros_like(arr)
    return (normalized * 255).astype(np.uint8)


def apply_zoom_to_array(arr: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Zoom into center of array by extracting central region.

    Args:
        arr: Input array
        zoom_factor: Zoom factor (e.g., 5.0 for 5x zoom)

    Returns:
        Zoomed array (same size as input, interpolated)
    """
    size = arr.shape[0]
    crop_size = int(size / zoom_factor)

    # Extract center region
    center = size // 2
    half_crop = crop_size // 2
    cropped = arr[center - half_crop:center + half_crop,
                  center - half_crop:center + half_crop]

    # Resize back to original size using PIL for better quality
    from scipy.ndimage import zoom
    zoomed = zoom(cropped, zoom_factor, order=1)

    # Ensure output is same size as input
    if zoomed.shape[0] != size:
        # Crop or pad to exact size
        if zoomed.shape[0] > size:
            excess = (zoomed.shape[0] - size) // 2
            zoomed = zoomed[excess:excess+size, excess:excess+size]
        else:
            pad = (size - zoomed.shape[0]) // 2
            zoomed = np.pad(zoomed, ((pad, size - zoomed.shape[0] - pad),
                                     (pad, size - zoomed.shape[1] - pad)),
                           mode='constant')

    return zoomed


def add_text_to_image(img: Image.Image, text: str, position: str = "top-right") -> Image.Image:
    """
    Add text label to image.

    Args:
        img: PIL Image
        text: Text to add
        position: Position ("top-right", "top-left", etc.)

    Returns:
        Image with text added
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fall back to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    margin = 5
    if position == "top-right":
        x = img.width - text_width - margin
        y = margin
    elif position == "top-left":
        x = margin
        y = margin
    else:
        x = margin
        y = margin

    # Draw text with outline for visibility
    outline_color = (80, 80, 80)  # Dark gray
    text_color = (200, 200, 200)  # Light gray

    # Draw outline
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    # Draw main text
    draw.text((x, y), text, font=font, fill=text_color)

    return img


def save_gif(frames: List[np.ndarray], path: Path, is_phase: bool = False, zoom_factor: float = None):
    """
    Save list of arrays as animated GIF.

    Args:
        frames: List of 2D arrays to save
        path: Output path
        is_phase: Whether this is phase data (affects colormap)
        zoom_factor: Optional zoom factor label to display (zoom is applied during intensity computation)
    """
    images = []
    for frame in frames:
        img_data = normalize_for_image(frame, is_phase)
        if is_phase:
            # Use twilight-like colormap for phase
            img = Image.fromarray(img_data, mode='L')
            img = img.convert('P')
            # Apply custom phase colormap (twilight-inspired)
            palette = []
            for i in range(256):
                t = i / 255.0
                # Twilight-like: purple -> blue -> white -> orange -> purple
                if t < 0.25:
                    r = int(100 + 155 * (t / 0.25))
                    g = int(50 * (t / 0.25))
                    b = int(150 + 105 * (t / 0.25))
                elif t < 0.5:
                    r = 255
                    g = int(50 + 205 * ((t - 0.25) / 0.25))
                    b = int(255 - 100 * ((t - 0.25) / 0.25))
                elif t < 0.75:
                    r = int(255 - 50 * ((t - 0.5) / 0.25))
                    g = int(255 - 50 * ((t - 0.5) / 0.25))
                    b = int(155 - 55 * ((t - 0.5) / 0.25))
                else:
                    r = int(205 - 105 * ((t - 0.75) / 0.25))
                    g = int(205 - 155 * ((t - 0.75) / 0.25))
                    b = int(100 + 50 * ((t - 0.75) / 0.25))
                palette.extend([r, g, b])
            img.putpalette(palette)
        else:
            # Hot colormap for intensity
            img = Image.fromarray(img_data, mode='L')
            img = img.convert('P')
            palette = []
            for i in range(256):
                t = i / 255.0
                r = int(min(255, t * 3 * 255))
                g = int(min(255, max(0, (t - 0.33) * 3 * 255)))
                b = int(min(255, max(0, (t - 0.67) * 3 * 255)))
                palette.extend([r, g, b])
            img.putpalette(palette)

            # Add zoom label if zoomed
            if zoom_factor is not None and zoom_factor > 1.0:
                img = img.convert('RGB')  # Convert to RGB for text overlay
                img = add_text_to_image(img, f"{zoom_factor:.1f}x", position="top-right")

        images.append(img)

    if len(images) == 1:
        images[0].save(path, save_all=False)
    else:
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            duration=GIF_DURATION_MS,
            loop=0
        )


def generate_sample(
    config: SampleConfig,
    output_dir: Path,
    input_amp: np.ndarray
) -> dict:
    """
    Generate a single sample with phase and intensity GIFs.

    Returns:
        Sample manifest entry dict
    """
    level_dir = output_dir / f"L{config.level}"
    level_dir.mkdir(parents=True, exist_ok=True)

    # Generate frames - pass zoom to generator if it accepts it
    import inspect
    sig = inspect.signature(config.generator)
    if 'zoom' in sig.parameters:
        frames = config.generator(input_amp, zoom=config.intensity_zoom or 1.0)
    else:
        frames = config.generator(input_amp)

    phase_frames = [f[0] for f in frames]
    intensity_frames = [f[1] for f in frames]

    # Save GIFs
    phase_path = level_dir / f"{config.id}_phase.gif"
    intensity_path = level_dir / f"{config.id}_intensity.gif"

    save_gif(phase_frames, phase_path, is_phase=True)
    save_gif(intensity_frames, intensity_path, is_phase=False, zoom_factor=config.intensity_zoom)

    return {
        "id": config.id,
        "level": config.level,
        "category": config.category,
        "name": config.name,
        "description": config.description,
        "phase_gif": f"assets/L{config.level}/{config.id}_phase.gif",
        "intensity_gif": f"assets/L{config.level}/{config.id}_intensity.gif",
        "parameters": config.parameters or {},
    }


# =============================================================================
# Level 1: Foundations - Sample Generators
# =============================================================================

def gen_uniform_phase(input_amp: np.ndarray):
    """Uniform (zero) phase - static baseline."""
    phase = create_uniform_phase(GRID_SIZE)
    intensity = compute_intensity(input_amp, phase)
    return [(phase, intensity)]


def gen_spot_sweep_x(input_amp: np.ndarray):
    """Single spot sweeping left to right."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_spot_target(GRID_SIZE, cx, 0, radius=4)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spot_sweep_y(input_amp: np.ndarray):
    """Single spot sweeping bottom to top."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_spot_target(GRID_SIZE, 0, cy, radius=4)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spot_circular(input_amp: np.ndarray):
    """Single spot moving in circle."""
    frames = []
    n_frames = 24
    radius = GRID_SIZE // 8
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)
        target = create_spot_target(GRID_SIZE, cx, cy, radius=4)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_gaussian_spot_sweep(input_amp: np.ndarray):
    """Gaussian spot (soft edges) sweeping."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_gaussian_spot_target(GRID_SIZE, cx, 0, sigma=6)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_slab_sweep_x(input_amp: np.ndarray):
    """Rectangular slab sweeping horizontally."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_rectangular_slab_target(GRID_SIZE, cx, 0, width=15, height=30)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_slab_sweep_y(input_amp: np.ndarray):
    """Rectangular slab sweeping vertically."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_rectangular_slab_target(GRID_SIZE, 0, cy, width=15, height=30)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_linear_ramp_x(input_amp: np.ndarray):
    """Linear phase ramp in X, kx sweep. 2x faster, 2x longer, 4x smaller end pitch."""
    frames = []
    n_frames = 32  # 2x more frames
    for i in range(n_frames):
        # End at 4x smaller pitch (4x larger k)
        kx = 2 * np.pi * i / (n_frames - 1) / (GRID_SIZE / 32)
        phase = create_linear_ramp(GRID_SIZE, kx=kx, ky=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_linear_ramp_y(input_amp: np.ndarray):
    """Linear phase ramp in Y, ky sweep. 2x faster, 2x longer, 4x smaller end pitch."""
    frames = []
    n_frames = 32  # 2x more frames
    for i in range(n_frames):
        # End at 4x smaller pitch (4x larger k)
        ky = 2 * np.pi * i / (n_frames - 1) / (GRID_SIZE / 32)
        phase = create_linear_ramp(GRID_SIZE, kx=0, ky=ky)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_linear_ramp_diagonal(input_amp: np.ndarray):
    """Linear phase ramp with rotating direction. 3x smaller pitch."""
    frames = []
    n_frames = 16
    # 3x smaller pitch = 3x larger k_mag
    k_mag = 2 * np.pi / (GRID_SIZE / 24)
    for i in range(n_frames):
        angle = np.pi / 4 * i / (n_frames - 1)  # 0 to 45 deg
        kx = k_mag * np.cos(angle)
        ky = k_mag * np.sin(angle)
        phase = create_linear_ramp(GRID_SIZE, kx=kx, ky=ky)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_positive(input_amp: np.ndarray):
    """Quadratic phase (positive curvature) sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        curvature = 0.5 + 3.0 * i / (n_frames - 1)
        phase = create_quadratic_phase(GRID_SIZE, curvature)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_negative(input_amp: np.ndarray):
    """Quadratic phase (negative curvature) sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        curvature = -0.5 - 3.0 * i / (n_frames - 1)
        phase = create_quadratic_phase(GRID_SIZE, curvature)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_cubic_x(input_amp: np.ndarray):
    """Cubic phase in X sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        coeff = 1.0 + 4.0 * i / (n_frames - 1)
        phase = create_cubic_phase(GRID_SIZE, coeff_x=coeff, coeff_y=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_cubic_y(input_amp: np.ndarray):
    """Cubic phase in Y sweep. 4x longer."""
    frames = []
    n_frames = 64  # 4x more frames
    for i in range(n_frames):
        coeff = 1.0 + 4.0 * i / (n_frames - 1)
        phase = create_cubic_phase(GRID_SIZE, coeff_x=0, coeff_y=coeff)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_coherent_aperture(input_amp: np.ndarray):
    """Variable coherent aperture - demonstrates spot size vs aperture."""
    frames = []
    n_frames = 16
    x = np.linspace(-GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    # Fixed random phase for outer region (seeded for consistency)
    rng = np.random.RandomState(42)
    random_phase = rng.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    for i in range(n_frames):
        # Radius from small to large (small aperture = large spot)
        # Start at 10px (2x smaller than before) up to 100px
        radius = 10 + 90 * i / (n_frames - 1)  # 10 to 100 pixels
        # Flat phase inside, random outside
        phase = np.where(R <= radius, 0.0, random_phase)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_soft_aperture(input_amp: np.ndarray):
    """Soft-edged aperture - demonstrates edge smoothness effect."""
    frames = []
    n_frames = 16
    x = np.linspace(-GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    rng = np.random.RandomState(42)
    random_phase = rng.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    radius = 30  # Fixed radius (2x smaller than before)
    for i in range(n_frames):
        # Edge softness from sharp to very soft
        softness = 1 + 30 * i / (n_frames - 1)  # 1 to 31 pixels
        # Soft transition using sigmoid-like function
        mask = 1.0 / (1.0 + np.exp((R - radius) / softness))
        # Blend flat phase (inside) with random phase (outside)
        phase = mask * 0.0 + (1 - mask) * random_phase
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


# =============================================================================
# L1: Additional Foundations - Line/Band Sweeps
# =============================================================================

def gen_narrow_line_sweep_x(input_amp: np.ndarray):
    """Narrow vertical line sweeping horizontally."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_line_target(GRID_SIZE, cx, 0, width=4, orientation='vertical')
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_narrow_line_sweep_y(input_amp: np.ndarray):
    """Narrow horizontal line sweeping vertically."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_line_target(GRID_SIZE, 0, cy, width=4, orientation='horizontal')
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_wide_band_sweep_x(input_amp: np.ndarray):
    """Wide vertical band sweeping horizontally."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_band_target(GRID_SIZE, cx, 0, width=35, orientation='vertical', profile='uniform')
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_wide_band_sweep_y(input_amp: np.ndarray):
    """Wide horizontal band sweeping vertically."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_band_target(GRID_SIZE, 0, cy, width=35, orientation='horizontal', profile='uniform')
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_half_sine_band_sweep_x(input_amp: np.ndarray):
    """Vertical band with half-sine profile sweeping horizontally."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cx = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_band_target(GRID_SIZE, cx, 0, width=35, orientation='vertical', profile='half_sine')
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_half_sine_band_sweep_y(input_amp: np.ndarray):
    """Horizontal band with half-sine profile sweeping vertically."""
    frames = []
    n_frames = 16
    max_offset = GRID_SIZE // 6
    for i in range(n_frames):
        cy = -max_offset + (2 * max_offset * i) / (n_frames - 1)
        target = create_band_target(GRID_SIZE, 0, cy, width=35, orientation='horizontal', profile='half_sine')
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


# =============================================================================
# L1: Additional Foundations - Disk/Ellipse Variations
# =============================================================================

def gen_disk_aspect_ratio_sweep(input_amp: np.ndarray):
    """Disk morphing from circle to ellipse - aspect ratio sweep."""
    frames = []
    n_frames = 20
    base_radius = 45
    for i in range(n_frames):
        aspect_ratio = 1.0 + 2.0 * i / (n_frames - 1)  # 1.0 to 3.0
        radius_x = base_radius
        radius_y = base_radius / aspect_ratio
        target = create_ellipse_target(GRID_SIZE, 0, 0, radius_x, radius_y, angle=0)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_ellipse_rotation(input_amp: np.ndarray):
    """Fixed ellipse rotating from 0 to 90 degrees."""
    frames = []
    n_frames = 20
    radius_x = 50
    radius_y = 20
    for i in range(n_frames):
        angle = np.pi / 2 * i / (n_frames - 1)  # 0 to 90 degrees
        target = create_ellipse_target(GRID_SIZE, 0, 0, radius_x, radius_y, angle=angle)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


# =============================================================================
# L1: Additional Foundations - Ring/Annulus Patterns
# =============================================================================

def gen_ring_aperture_radius_sweep(input_amp: np.ndarray):
    """Ring aperture - radius increases while keeping width constant. Uses random phase outside ring."""
    frames = []
    n_frames = 16
    ring_width = 15  # Fixed width
    # Fixed random phase for outside region (seeded for consistency)
    rng = np.random.RandomState(43)
    random_phase = rng.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    for i in range(n_frames):
        mean_radius = 20 + 50 * i / (n_frames - 1)  # 20 to 70 pixels
        inner_radius = mean_radius - ring_width / 2
        outer_radius = mean_radius + ring_width / 2
        aperture_mask = create_ring_aperture_phase(GRID_SIZE, inner_radius, outer_radius)
        # Flat phase inside ring, random phase outside (scrambles light)
        phase = np.where(aperture_mask > 0.5, 0.0, random_phase)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_ring_aperture_width_sweep(input_amp: np.ndarray):
    """Ring aperture - width increases, mean radius fixed. Uses random phase outside ring."""
    frames = []
    n_frames = 16
    mean_radius = 50
    # Fixed random phase for outside region (seeded for consistency)
    rng = np.random.RandomState(44)
    random_phase = rng.uniform(-np.pi, np.pi, (GRID_SIZE, GRID_SIZE))

    for i in range(n_frames):
        width = 5 + 35 * i / (n_frames - 1)  # 5 to 40 pixels
        inner_radius = mean_radius - width / 2
        outer_radius = mean_radius + width / 2
        aperture_mask = create_ring_aperture_phase(GRID_SIZE, inner_radius, outer_radius)
        # Flat phase inside ring, random phase outside (scrambles light)
        phase = np.where(aperture_mask > 0.5, 0.0, random_phase)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_ring_target_radius_sweep(input_amp: np.ndarray):
    """Ring target intensity - radius sweeps outward."""
    frames = []
    n_frames = 16
    width = 8
    for i in range(n_frames):
        mean_radius = 15 + 45 * i / (n_frames - 1)  # 15 to 60 pixels
        inner_radius = mean_radius - width / 2
        outer_radius = mean_radius + width / 2
        target = create_ring_target(GRID_SIZE, 0, 0, inner_radius, outer_radius)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_ring_target_width_sweep(input_amp: np.ndarray):
    """Ring target intensity - width increases, radius fixed."""
    frames = []
    n_frames = 16
    mean_radius = 40
    for i in range(n_frames):
        width = 5 + 35 * i / (n_frames - 1)  # 5 to 40 pixels
        inner_radius = mean_radius - width / 2
        outer_radius = mean_radius + width / 2
        target = create_ring_target(GRID_SIZE, 0, 0, inner_radius, outer_radius)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


# =============================================================================
# L1: Additional Foundations - Cylindrical Lens Variations
# =============================================================================

def gen_quadratic_cylindrical_x(input_amp: np.ndarray):
    """Cylindrical lens in X - creates line focus."""
    frames = []
    n_frames = 32
    for i in range(n_frames):
        curvature = -10 + 20 * i / (n_frames - 1)  # -10 to +10 (much stronger)
        phase = create_cylindrical_lens_phase(GRID_SIZE, curvature, axis='x')
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_quadratic_cylindrical_y(input_amp: np.ndarray):
    """Cylindrical lens in Y - line focus rotated 90 degrees."""
    frames = []
    n_frames = 32
    for i in range(n_frames):
        curvature = -10 + 20 * i / (n_frames - 1)  # -10 to +10 (much stronger)
        phase = create_cylindrical_lens_phase(GRID_SIZE, curvature, axis='y')
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


# =============================================================================
# L1: Additional Foundations - Multi-Spot Patterns
# =============================================================================

def gen_two_spots_separation_sweep(input_amp: np.ndarray):
    """Two spots - separation increases."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        separation = 10 + 60 * i / (n_frames - 1)  # 10 to 70 pixels
        target = create_two_spots_target(GRID_SIZE, separation, angle=0)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spot_size_sweep(input_amp: np.ndarray):
    """Single centered spot - radius increases."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        radius = 3 + 22 * i / (n_frames - 1)  # 3 to 25 pixels
        target = create_spot_target(GRID_SIZE, 0, 0, radius=int(radius))
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


# =============================================================================
# L2: Periodic Structures Frame Generators
# =============================================================================

def gen_binary_grating_vertical(input_amp: np.ndarray):
    """Binary grating vertical, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_binary_grating(GRID_SIZE, period=period, angle=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_binary_grating_horizontal(input_amp: np.ndarray):
    """Binary grating horizontal, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_binary_grating(GRID_SIZE, period=period, angle=np.pi/2)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_binary_grating_rotated(input_amp: np.ndarray):
    """Binary grating with rotating angle."""
    frames = []
    n_frames = 16
    period = 24  # Smaller pitch for rotation demo
    for i in range(n_frames):
        angle = np.pi / 2 * i / (n_frames - 1)  # 0 to 90 deg
        phase = create_binary_grating(GRID_SIZE, period=period, angle=angle)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_sinusoidal_grating(input_amp: np.ndarray):
    """Sinusoidal grating, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_sinusoidal_grating(GRID_SIZE, period=period, angle=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_blazed_grating(input_amp: np.ndarray):
    """Blazed grating, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_blazed_grating(GRID_SIZE, period=period, angle=0)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_checkerboard(input_amp: np.ndarray):
    """Checkerboard pattern, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 56 * i / (n_frames - 1)  # 64 to 8 pixels
        phase = create_checkerboard(GRID_SIZE, period=period)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_crossed_gratings(input_amp: np.ndarray):
    """Crossed gratings with rotating angle."""
    frames = []
    n_frames = 16
    period = 16  # 2x smaller pitch
    for i in range(n_frames):
        angle = np.pi / 4 * i / (n_frames - 1)  # 0 to 45 deg
        phase = create_crossed_gratings(GRID_SIZE, period=period, angle=angle)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_multi_frequency_grating(input_amp: np.ndarray):
    """Multi-frequency grating, ratio sweep."""
    frames = []
    n_frames = 16
    period1 = 32
    for i in range(n_frames):
        ratio = 2 + 3 * i / (n_frames - 1)  # period ratio from 2 to 5
        period2 = period1 / ratio
        phase = create_multi_frequency_grating(GRID_SIZE, period1=period1, period2=period2)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


# =============================================================================
# L2 Extended: Advanced Periodic Structures
# =============================================================================

def gen_concentric_rings_binary(input_amp: np.ndarray):
    """Concentric binary rings, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 40 - 30 * i / (n_frames - 1)  # 40 to 10 pixels
        phase = create_concentric_rings_binary(GRID_SIZE, period=period)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_concentric_rings_sinusoidal(input_amp: np.ndarray):
    """Concentric sinusoidal rings (Fresnel zone plate-like), period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 40 - 30 * i / (n_frames - 1)  # 40 to 10 pixels
        phase = create_concentric_rings_sinusoidal(GRID_SIZE, period=period)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_radial_sectors(input_amp: np.ndarray):
    """Radial sectors (pizza slices), number of sectors increases."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        n_sectors = 4 + int(28 * i / (n_frames - 1))  # 4 to 32 sectors
        phase = create_radial_sectors(GRID_SIZE, n_sectors=n_sectors)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_spiral_grating(input_amp: np.ndarray):
    """Spiral grating, pitch decreases."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        pitch = 50 - 35 * i / (n_frames - 1)  # 50 to 15 pixels
        phase = create_spiral_grating(GRID_SIZE, pitch=pitch, n_arms=1)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_hexagonal_lattice(input_amp: np.ndarray):
    """Hexagonal lattice, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 60 - 40 * i / (n_frames - 1)  # 60 to 20 pixels
        phase = create_hexagonal_lattice(GRID_SIZE, period=period)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_triangular_lattice(input_amp: np.ndarray):
    """Triangular lattice, period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 60 - 40 * i / (n_frames - 1)  # 60 to 20 pixels
        phase = create_triangular_lattice(GRID_SIZE, period=period)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_grating_with_defect(input_amp: np.ndarray):
    """Binary grating with phase defect moving across."""
    frames = []
    n_frames = 16
    period = 20
    for i in range(n_frames):
        defect_pos = -GRID_SIZE/2 + GRID_SIZE * i / (n_frames - 1)  # Move from left to right
        phase = create_grating_with_defect(GRID_SIZE, period=period, defect_position=defect_pos)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_chirped_grating(input_amp: np.ndarray):
    """Chirped grating, chirp rate increases."""
    frames = []
    n_frames = 16
    period_start = 40
    for i in range(n_frames):
        period_end = 40 - 30 * i / (n_frames - 1)  # 40 to 10 pixels (increasing chirp)
        phase = create_chirped_grating(GRID_SIZE, period_start=period_start, period_end=period_end)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_duty_cycle_sweep(input_amp: np.ndarray):
    """Binary grating with duty cycle sweep."""
    frames = []
    n_frames = 16
    period = 24
    for i in range(n_frames):
        duty_cycle = 0.2 + 0.6 * i / (n_frames - 1)  # 0.2 to 0.8
        phase = create_duty_cycle_grating(GRID_SIZE, period=period, duty_cycle=duty_cycle)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_amplitude_modulated_grating(input_amp: np.ndarray):
    """Grating with Gaussian envelope, envelope width decreases."""
    frames = []
    n_frames = 16
    period = 20
    for i in range(n_frames):
        sigma = GRID_SIZE/2 - GRID_SIZE/3 * i / (n_frames - 1)  # Envelope narrows
        phase = create_amplitude_modulated_grating(GRID_SIZE, period=period, envelope_sigma=sigma)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_blazed_grating_angle_sweep(input_amp: np.ndarray):
    """Blazed grating with varying blaze angle."""
    frames = []
    n_frames = 16
    period = 24
    for i in range(n_frames):
        blaze_fraction = 0.1 + 0.8 * i / (n_frames - 1)  # 0.1 to 0.9
        phase = create_blazed_grating_variable(GRID_SIZE, period=period, blaze_fraction=blaze_fraction)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


def gen_rings_and_sectors(input_amp: np.ndarray):
    """Combined concentric rings and radial sectors pattern."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        ring_period = 40 - 30 * i / (n_frames - 1)  # 40 to 10 pixels (same as standalone rings)
        n_sectors = 4 + int(28 * i / (n_frames - 1))  # 4 to 32 sectors
        phase = create_rings_and_sectors(GRID_SIZE, ring_period=ring_period, n_sectors=n_sectors)
        intensity = compute_intensity(input_amp, phase)
        frames.append((phase, intensity))
    return frames


# =============================================================================
# Sample Configuration Registry
# =============================================================================

L1_SAMPLES = [
    SampleConfig(
        id="uniform_phase",
        level=1,
        category="foundations",
        name="Uniform Phase",
        description="Zero phase everywhere - baseline reference showing unmodified Gaussian",
        generator=gen_uniform_phase,
    ),
    SampleConfig(
        id="spot_sweep_x",
        level=1,
        category="foundations",
        name="Single Spot Sweep (X)",
        description="Spot moves left to right - observe phase ramp tilting",
        generator=gen_spot_sweep_x,
    ),
    SampleConfig(
        id="spot_sweep_y",
        level=1,
        category="foundations",
        name="Single Spot Sweep (Y)",
        description="Spot moves bottom to top - vertical phase ramp",
        generator=gen_spot_sweep_y,
    ),
    SampleConfig(
        id="spot_circular",
        level=1,
        category="foundations",
        name="Single Spot Circular",
        description="Spot moves in circle - phase ramp rotates",
        generator=gen_spot_circular,
    ),
    SampleConfig(
        id="gaussian_spot_sweep",
        level=1,
        category="foundations",
        name="Gaussian Spot Sweep",
        description="Soft Gaussian spot - no sinc ringing due to smooth edges",
        generator=gen_gaussian_spot_sweep,
    ),
    SampleConfig(
        id="slab_sweep_x",
        level=1,
        category="foundations",
        name="Rectangular Slab Sweep (X)",
        description="Rectangle moves horizontally - observe sinc envelope",
        generator=gen_slab_sweep_x,
    ),
    SampleConfig(
        id="slab_sweep_y",
        level=1,
        category="foundations",
        name="Rectangular Slab Sweep (Y)",
        description="Rectangle moves vertically - sinc in orthogonal direction",
        generator=gen_slab_sweep_y,
    ),
    SampleConfig(
        id="linear_ramp_x",
        level=1,
        category="foundations",
        name="Linear Ramp (X)",
        description="Phase gradient in X shifts beam horizontally in Fourier plane",
        generator=gen_linear_ramp_x,
    ),
    SampleConfig(
        id="linear_ramp_y",
        level=1,
        category="foundations",
        name="Linear Ramp (Y)",
        description="Phase gradient in Y shifts beam vertically in Fourier plane",
        generator=gen_linear_ramp_y,
    ),
    SampleConfig(
        id="linear_ramp_diagonal",
        level=1,
        category="foundations",
        name="Linear Ramp (Diagonal)",
        description="Phase ramp direction rotates from X toward diagonal",
        generator=gen_linear_ramp_diagonal,
    ),
    SampleConfig(
        id="quadratic_positive",
        level=1,
        category="foundations",
        name="Quadratic Phase (+)",
        description="Positive curvature (converging lens) - ring pattern expands",
        generator=gen_quadratic_positive,
    ),
    SampleConfig(
        id="quadratic_negative",
        level=1,
        category="foundations",
        name="Quadratic Phase (-)",
        description="Negative curvature (diverging lens) - ring pattern",
        generator=gen_quadratic_negative,
    ),
    SampleConfig(
        id="cubic_x",
        level=1,
        category="foundations",
        name="Cubic Phase (X)",
        description="Cubic phase in X - Airy-like asymmetric pattern",
        generator=gen_cubic_x,
    ),
    SampleConfig(
        id="cubic_y",
        level=1,
        category="foundations",
        name="Cubic Phase (Y)",
        description="Cubic phase in Y - rotated Airy pattern",
        generator=gen_cubic_y,
    ),
    SampleConfig(
        id="coherent_aperture",
        level=1,
        category="foundations",
        name="Coherent Aperture Size",
        description="Varying coherent region size - smaller aperture = larger spot (Fourier relationship)",
        generator=gen_coherent_aperture,
    ),
    SampleConfig(
        id="soft_aperture",
        level=1,
        category="foundations",
        name="Soft Aperture Edge",
        description="Edge smoothness sweep - softer edges reduce ringing/sidelobes",
        generator=gen_soft_aperture,
    ),
    # New additions - Line/Band sweeps
    SampleConfig(
        id="narrow_line_sweep_x",
        level=1,
        category="foundations",
        name="Narrow Line Sweep (X)",
        description="Thin vertical line sweeping horizontally - sharp sinc pattern",
        generator=gen_narrow_line_sweep_x,
    ),
    SampleConfig(
        id="narrow_line_sweep_y",
        level=1,
        category="foundations",
        name="Narrow Line Sweep (Y)",
        description="Thin horizontal line sweeping vertically - sharp sinc pattern",
        generator=gen_narrow_line_sweep_y,
    ),
    SampleConfig(
        id="wide_band_sweep_x",
        level=1,
        category="foundations",
        name="Wide Band Sweep (X)",
        description="Thick vertical band sweeping horizontally - broader sinc envelope",
        generator=gen_wide_band_sweep_x,
    ),
    SampleConfig(
        id="wide_band_sweep_y",
        level=1,
        category="foundations",
        name="Wide Band Sweep (Y)",
        description="Thick horizontal band sweeping vertically - broader sinc envelope",
        generator=gen_wide_band_sweep_y,
    ),
    SampleConfig(
        id="half_sine_band_sweep_x",
        level=1,
        category="foundations",
        name="Half-Sine Band Sweep (X)",
        description="Vertical band with half-sine profile - reduced sidelobes vs rectangular",
        generator=gen_half_sine_band_sweep_x,
    ),
    SampleConfig(
        id="half_sine_band_sweep_y",
        level=1,
        category="foundations",
        name="Half-Sine Band Sweep (Y)",
        description="Horizontal band with half-sine profile - smoother than rectangular",
        generator=gen_half_sine_band_sweep_y,
    ),
    # Disk/Ellipse variations
    SampleConfig(
        id="disk_aspect_ratio_sweep",
        level=1,
        category="foundations",
        name="Disk Aspect Ratio Sweep",
        description="Disk morphs from circle to ellipse - Airy pattern deforms",
        generator=gen_disk_aspect_ratio_sweep,
    ),
    SampleConfig(
        id="ellipse_rotation",
        level=1,
        category="foundations",
        name="Ellipse Rotation",
        description="Fixed ellipse rotates 0° to 90° - intensity pattern rotates",
        generator=gen_ellipse_rotation,
    ),
    # Ring/Annulus patterns
    SampleConfig(
        id="ring_aperture_radius_sweep",
        level=1,
        category="foundations",
        name="Ring Aperture Radius Sweep",
        description="Annular aperture - outer radius increases, shows Bessel-like patterns",
        generator=gen_ring_aperture_radius_sweep,
    ),
    SampleConfig(
        id="ring_aperture_width_sweep",
        level=1,
        category="foundations",
        name="Ring Aperture Width Sweep",
        description="Annular aperture - ring width increases from thin to thick",
        generator=gen_ring_aperture_width_sweep,
    ),
    SampleConfig(
        id="ring_target_radius_sweep",
        level=1,
        category="foundations",
        name="Ring Target Radius Sweep",
        description="Ring-shaped target - radius sweeps outward",
        generator=gen_ring_target_radius_sweep,
    ),
    SampleConfig(
        id="ring_target_width_sweep",
        level=1,
        category="foundations",
        name="Ring Target Width Sweep",
        description="Ring-shaped target - width increases, radius fixed",
        generator=gen_ring_target_width_sweep,
    ),
    # Cylindrical lens variations
    SampleConfig(
        id="quadratic_cylindrical_x",
        level=1,
        category="foundations",
        name="Cylindrical Lens (X)",
        description="Quadratic phase in X only - creates line focus in Fourier plane",
        generator=gen_quadratic_cylindrical_x,
    ),
    SampleConfig(
        id="quadratic_cylindrical_y",
        level=1,
        category="foundations",
        name="Cylindrical Lens (Y)",
        description="Quadratic phase in Y only - line focus rotated 90°",
        generator=gen_quadratic_cylindrical_y,
    ),
    # Multi-spot patterns
    SampleConfig(
        id="two_spots_separation_sweep",
        level=1,
        category="foundations",
        name="Two Spots Separation Sweep",
        description="Two spots - separation increases from close to far apart",
        generator=gen_two_spots_separation_sweep,
    ),
    SampleConfig(
        id="spot_size_sweep",
        level=1,
        category="foundations",
        name="Spot Size Sweep",
        description="Single centered spot - radius increases, inverse relationship in Fourier plane",
        generator=gen_spot_size_sweep,
    ),
]

L2_SAMPLES = [
    SampleConfig(
        id="binary_grating_vertical",
        level=2,
        category="periodic",
        name="Binary Grating (Vertical)",
        description="Vertical binary grating - period decreases, diffraction orders separate",
        generator=gen_binary_grating_vertical,
    ),
    SampleConfig(
        id="binary_grating_horizontal",
        level=2,
        category="periodic",
        name="Binary Grating (Horizontal)",
        description="Horizontal binary grating - diffraction in vertical direction",
        generator=gen_binary_grating_horizontal,
    ),
    SampleConfig(
        id="binary_grating_rotated",
        level=2,
        category="periodic",
        name="Binary Grating (Rotating)",
        description="Binary grating rotates 0° to 90° - watch diffraction spots rotate",
        generator=gen_binary_grating_rotated,
    ),
    SampleConfig(
        id="sinusoidal_grating",
        level=2,
        category="periodic",
        name="Sinusoidal Grating",
        description="Sinusoidal phase grating - only ±1 orders, no higher harmonics",
        generator=gen_sinusoidal_grating,
    ),
    SampleConfig(
        id="blazed_grating",
        level=2,
        category="periodic",
        name="Blazed Grating",
        description="Sawtooth phase grating - asymmetric diffraction, power in +1 order",
        generator=gen_blazed_grating,
    ),
    SampleConfig(
        id="checkerboard",
        level=2,
        category="periodic",
        name="Checkerboard",
        description="2D periodic pattern - 4-spot diffraction pattern",
        generator=gen_checkerboard,
    ),
    SampleConfig(
        id="crossed_gratings",
        level=2,
        category="periodic",
        name="Crossed Gratings",
        description="Two perpendicular gratings - XOR pattern creates grid diffraction",
        generator=gen_crossed_gratings,
    ),
    SampleConfig(
        id="multi_frequency_grating",
        level=2,
        category="periodic",
        name="Multi-Frequency Grating",
        description="Superposition of two frequencies - multiple diffraction orders",
        generator=gen_multi_frequency_grating,
    ),
    # Radial/Circular Patterns
    SampleConfig(
        id="concentric_rings_binary",
        level=2,
        category="periodic",
        name="Concentric Rings (Binary)",
        description="Binary phase rings - radial diffraction pattern, period decreases",
        generator=gen_concentric_rings_binary,
    ),
    SampleConfig(
        id="concentric_rings_sinusoidal",
        level=2,
        category="periodic",
        name="Concentric Rings (Sinusoidal)",
        description="Smooth sinusoidal rings (Fresnel zone plate-like) - radial focusing effect",
        generator=gen_concentric_rings_sinusoidal,
    ),
    SampleConfig(
        id="radial_sectors",
        level=2,
        category="periodic",
        name="Radial Sectors",
        description="Pizza slice pattern - number of sectors increases, azimuthal diffraction",
        generator=gen_radial_sectors,
    ),
    SampleConfig(
        id="spiral_grating",
        level=2,
        category="periodic",
        name="Spiral Grating",
        description="Archimedean spiral - pitch decreases, creates orbital angular momentum",
        generator=gen_spiral_grating,
    ),
    # Lattice Patterns
    SampleConfig(
        id="hexagonal_lattice",
        level=2,
        category="periodic",
        name="Hexagonal Lattice",
        description="Honeycomb pattern - 6-fold symmetric diffraction, period sweep",
        generator=gen_hexagonal_lattice,
    ),
    SampleConfig(
        id="triangular_lattice",
        level=2,
        category="periodic",
        name="Triangular Lattice",
        description="Triangular tiling - 3-fold symmetric diffraction pattern",
        generator=gen_triangular_lattice,
    ),
    # Advanced Grating Variations
    SampleConfig(
        id="grating_with_defect",
        level=2,
        category="periodic",
        name="Grating with Defect",
        description="Binary grating with phase defect - observe defect mode in diffraction",
        generator=gen_grating_with_defect,
    ),
    SampleConfig(
        id="chirped_grating",
        level=2,
        category="periodic",
        name="Chirped Grating",
        description="Linearly varying period - spatially dispersive diffraction",
        generator=gen_chirped_grating,
    ),
    SampleConfig(
        id="duty_cycle_sweep",
        level=2,
        category="periodic",
        name="Duty Cycle Sweep",
        description="Binary grating with varying duty cycle - relative order intensities change",
        generator=gen_duty_cycle_sweep,
    ),
    SampleConfig(
        id="amplitude_modulated_grating",
        level=2,
        category="periodic",
        name="Amplitude Modulated Grating",
        description="Grating with Gaussian envelope - localized diffraction pattern",
        generator=gen_amplitude_modulated_grating,
    ),
    SampleConfig(
        id="blazed_grating_angle_sweep",
        level=2,
        category="periodic",
        name="Blazed Grating (Angle Sweep)",
        description="Blazed grating with varying blaze angle - efficiency shifts between orders",
        generator=gen_blazed_grating_angle_sweep,
    ),
    SampleConfig(
        id="rings_and_sectors",
        level=2,
        category="periodic",
        name="Rings and Sectors",
        description="Combined concentric rings and radial sectors - polar coordinate grid pattern",
        generator=gen_rings_and_sectors,
    ),
]


# =============================================================================
# Level 3 Generator Functions: Discrete Spot Arrays
# =============================================================================

def gen_grid_2x2_spacing_sweep(input_amp: np.ndarray):
    """2x2 grid with spacing sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        spacing = 20 + 40 * i / (n_frames - 1)  # 20 to 60 px
        target = create_grid_spots(GRID_SIZE, 2, 2, spacing)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_grid_3x3_spacing_sweep(input_amp: np.ndarray):
    """3x3 grid with spacing sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        spacing = 15 + 30 * i / (n_frames - 1)  # 15 to 45 px
        target = create_grid_spots(GRID_SIZE, 3, 3, spacing)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_grid_4x4_uniform(input_amp: np.ndarray):
    """4x4 grid with fixed spacing."""
    target = create_grid_spots(GRID_SIZE, 4, 4, 20)
    result = standard_gs(input_amp, target, GS_ITERATIONS)
    return [(result.phase_mask, result.reconstructed)]


def gen_grid_2x3_rectangular(input_amp: np.ndarray):
    """2x3 rectangular grid with aspect ratio sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        spacing_x = 30
        spacing_y = 20 + 30 * i / (n_frames - 1)  # 20 to 50 px
        target = create_grid_spots(GRID_SIZE, 2, 3, spacing_x, spacing_y)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_grid_3x3_rotation(input_amp: np.ndarray):
    """3x3 grid rotation."""
    frames = []
    n_frames = 24
    spacing = 25
    for i in range(n_frames):
        angle = np.pi / 4 * i / (n_frames - 1)  # 0 to 45 degrees
        # Generate rotated grid positions
        positions = []
        for row in range(3):
            for col in range(3):
                x = (col - 1) * spacing
                y = (row - 1) * spacing
                # Rotate
                xr = x * np.cos(angle) - y * np.sin(angle)
                yr = x * np.sin(angle) + y * np.cos(angle)
                positions.append((xr, yr))
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_grid_size_progression(input_amp: np.ndarray):
    """Grid size progression from 2x2 to 3x3 to 4x4."""
    frames = []
    spacing = 25

    # 2x2 for 8 frames
    for _ in range(8):
        target = create_grid_spots(GRID_SIZE, 2, 2, spacing)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))

    # 3x3 for 8 frames
    for _ in range(8):
        target = create_grid_spots(GRID_SIZE, 3, 3, spacing)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))

    # 4x4 for 8 frames
    for _ in range(8):
        target = create_grid_spots(GRID_SIZE, 4, 4, spacing)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))

    return frames


def gen_spots_brightness_gradient_x(input_amp: np.ndarray):
    """1x4 spots with horizontal brightness gradient."""
    frames = []
    n_frames = 16
    spacing = 35
    for i in range(n_frames):
        # Rotate gradient direction
        angle = np.pi / 2 * i / (n_frames - 1)  # 0 to 90 degrees
        positions = []
        brightness = []
        for j in range(4):
            x = (j - 1.5) * spacing
            y = 0
            # Rotate
            xr = x * np.cos(angle) - y * np.sin(angle)
            yr = x * np.sin(angle) + y * np.cos(angle)
            positions.append((xr, yr))
            brightness.append(j + 1)  # 1, 2, 3, 4
        target = create_spots_at_positions(GRID_SIZE, positions, brightness=brightness)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_brightness_ratio_2x2(input_amp: np.ndarray):
    """2x2 grid with one corner brightening."""
    frames = []
    n_frames = 16
    spacing = 30
    for i in range(n_frames):
        ratio = 1 + 3 * i / (n_frames - 1)  # 1 to 4
        brightness = np.array([[ratio, 1], [1, 1]])
        target = create_grid_spots(GRID_SIZE, 2, 2, spacing, brightness=brightness)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_checkerboard_brightness(input_amp: np.ndarray):
    """3x3 grid with checkerboard brightness pattern."""
    frames = []
    n_frames = 16
    spacing = 25
    for i in range(n_frames):
        ratio = 1 + 2 * i / (n_frames - 1)  # 1 to 3
        brightness = np.array([
            [ratio, 1, ratio],
            [1, ratio, 1],
            [ratio, 1, ratio]
        ])
        target = create_grid_spots(GRID_SIZE, 3, 3, spacing, brightness=brightness)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_center_bright(input_amp: np.ndarray):
    """3x3 grid with center spot brightening."""
    frames = []
    n_frames = 16
    spacing = 25
    for i in range(n_frames):
        ratio = 1 + 4 * i / (n_frames - 1)  # 1 to 5
        brightness = np.ones((3, 3))
        brightness[1, 1] = ratio
        target = create_grid_spots(GRID_SIZE, 3, 3, spacing, brightness=brightness)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_diagonal_line(input_amp: np.ndarray):
    """5 spots along diagonal with spacing sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        base_spacing = 15 + 25 * i / (n_frames - 1)  # 15 to 40 px
        positions = []
        for j in range(5):
            offset = (j - 2) * base_spacing
            positions.append((offset, offset))
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_L_shape(input_amp: np.ndarray):
    """6 spots forming L shape with rotation."""
    frames = []
    n_frames = 16
    spacing = 25
    for i in range(n_frames):
        angle = np.pi / 2 * i / (n_frames - 1)  # 0 to 90 degrees
        positions = []
        # Horizontal arm: 3 spots
        for j in range(3):
            x = j * spacing
            y = 0
            xr = x * np.cos(angle) - y * np.sin(angle)
            yr = x * np.sin(angle) + y * np.cos(angle)
            positions.append((xr, yr))
        # Vertical arm: 3 spots (excluding corner)
        for j in range(1, 3):
            x = 0
            y = j * spacing
            xr = x * np.cos(angle) - y * np.sin(angle)
            yr = x * np.sin(angle) + y * np.cos(angle)
            positions.append((xr, yr))
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_circle_arrangement(input_amp: np.ndarray):
    """8 spots arranged in circle with radius sweep."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        radius = 20 + 40 * i / (n_frames - 1)  # 20 to 60 px
        positions = []
        for j in range(8):
            angle = 2 * np.pi * j / 8
            cx = radius * np.cos(angle)
            cy = radius * np.sin(angle)
            positions.append((cx, cy))
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_asymmetric_cross(input_amp: np.ndarray):
    """5 spots in cross pattern with asymmetric arm lengths."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        arm_length = 15 + 30 * i / (n_frames - 1)  # 15 to 45 px
        positions = [
            (0, 0),  # center
            (arm_length, 0),  # right
            (-arm_length * 0.7, 0),  # left (shorter)
            (0, arm_length * 0.8),  # top
            (0, -arm_length * 0.6),  # bottom (shorter)
        ]
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_triangle_vertices(input_amp: np.ndarray):
    """3 spots at triangle corners with rotation and scale."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames  # Full rotation
        radius = 30 + 20 * np.sin(2 * np.pi * i / n_frames)  # 30 ± 20 px
        positions = []
        for j in range(3):
            theta = angle + 2 * np.pi * j / 3
            cx = radius * np.cos(theta)
            cy = radius * np.sin(theta)
            positions.append((cx, cy))
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_random_4(input_amp: np.ndarray):
    """4 spots at random positions - different configurations."""
    frames = []
    n_frames = 8
    for i in range(n_frames):
        positions = create_random_spot_positions(4, GRID_SIZE, min_separation=20, seed=100+i)
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_random_6(input_amp: np.ndarray):
    """6 spots at random positions - different configurations."""
    frames = []
    n_frames = 8
    for i in range(n_frames):
        positions = create_random_spot_positions(6, GRID_SIZE, min_separation=18, seed=200+i)
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_random_to_grid(input_amp: np.ndarray):
    """4 spots morphing from random positions to 2x2 grid."""
    frames = []
    n_frames = 24

    # Random start positions
    random_pos = create_random_spot_positions(4, GRID_SIZE, min_separation=25, seed=300)

    # Grid end positions
    spacing = 30
    grid_pos = [
        (-spacing/2, -spacing/2),
        (spacing/2, -spacing/2),
        (-spacing/2, spacing/2),
        (spacing/2, spacing/2),
    ]

    for i in range(n_frames):
        t = i / (n_frames - 1)  # 0 to 1
        positions = []
        for (rx, ry), (gx, gy) in zip(random_pos, grid_pos):
            cx = rx + t * (gx - rx)
            cy = ry + t * (gy - ry)
            positions.append((cx, cy))
        target = create_spots_at_positions(GRID_SIZE, positions)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_varying_size_3x3(input_amp: np.ndarray):
    """3x3 grid with varying spot sizes."""
    frames = []
    n_frames = 16
    spacing = 25
    for i in range(n_frames):
        size_ratio = 1 + 1.5 * i / (n_frames - 1)  # 1 to 2.5
        positions = []
        radii = []
        for row in range(3):
            for col in range(3):
                x = (col - 1) * spacing
                y = (row - 1) * spacing
                positions.append((x, y))
                # Center spot larger, corners smaller
                dist_from_center = np.sqrt((row - 1)**2 + (col - 1)**2)
                if dist_from_center == 0:
                    radii.append(int(6 * size_ratio))
                elif dist_from_center < 1.5:
                    radii.append(int(4.5 * size_ratio))
                else:
                    radii.append(int(3 * size_ratio))

        # Create target with varying sizes
        target = np.zeros((GRID_SIZE, GRID_SIZE))
        x = np.arange(GRID_SIZE) - GRID_SIZE // 2
        X, Y = np.meshgrid(x, x)
        for (cx, cy), r in zip(positions, radii):
            r2 = (X - cx)**2 + (Y - cy)**2
            mask = r2 <= r**2
            target[mask] = 1.0

        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


def gen_spots_dense_cluster(input_amp: np.ndarray):
    """9 spots tightly clustered with cluster size varying."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        spacing = 8 + 17 * i / (n_frames - 1)  # 8 to 25 px
        target = create_grid_spots(GRID_SIZE, 3, 3, spacing, radius=3)
        result = standard_gs(input_amp, target, GS_ITERATIONS)
        frames.append((result.phase_mask, result.reconstructed))
    return frames


# =============================================================================
# Level 3 Sample Configurations
# =============================================================================

L3_SAMPLES = [
    # Regular Grid Arrays
    SampleConfig(
        id="grid_2x2_spacing_sweep",
        level=3,
        category="spot_arrays",
        name="2x2 Grid Spacing Sweep",
        description="Four spots in square grid - spacing increases, observe phase periodicity scaling",
        generator=gen_grid_2x2_spacing_sweep,
    ),
    SampleConfig(
        id="grid_3x3_spacing_sweep",
        level=3,
        category="spot_arrays",
        name="3x3 Grid Spacing Sweep",
        description="Nine spots in 3x3 grid - denser array shows finer phase structure",
        generator=gen_grid_3x3_spacing_sweep,
    ),
    SampleConfig(
        id="grid_4x4_uniform",
        level=3,
        category="spot_arrays",
        name="4x4 Grid Uniform",
        description="Sixteen spots in 4x4 grid - demonstrates GS handling of many spots",
        generator=gen_grid_4x4_uniform,
    ),
    SampleConfig(
        id="grid_2x3_rectangular",
        level=3,
        category="spot_arrays",
        name="2x3 Rectangular Grid",
        description="Non-square grid (2 rows, 3 columns) - asymmetric spacing in phase",
        generator=gen_grid_2x3_rectangular,
    ),
    SampleConfig(
        id="grid_3x3_rotation",
        level=3,
        category="spot_arrays",
        name="3x3 Grid Rotation",
        description="3x3 grid rotates 0 to 45 degrees - entire phase pattern rotates with spots",
        generator=gen_grid_3x3_rotation,
    ),
    SampleConfig(
        id="grid_size_progression",
        level=3,
        category="spot_arrays",
        name="Grid Size Progression",
        description="Grid grows from 2x2 to 3x3 to 4x4 - complexity increases",
        generator=gen_grid_size_progression,
    ),
    # Non-Uniform Brightness
    SampleConfig(
        id="spots_brightness_gradient_x",
        level=3,
        category="spot_arrays",
        name="Brightness Gradient (Rotating)",
        description="Four spots with brightness gradient 1:2:3:4 - gradient direction rotates",
        generator=gen_spots_brightness_gradient_x,
    ),
    SampleConfig(
        id="spots_brightness_ratio_2x2",
        level=3,
        category="spot_arrays",
        name="2x2 Brightness Ratio Sweep",
        description="Four spots, one corner brightens - ratio sweeps from 1:1:1:1 to 4:1:1:1",
        generator=gen_spots_brightness_ratio_2x2,
    ),
    SampleConfig(
        id="spots_checkerboard_brightness",
        level=3,
        category="spot_arrays",
        name="Checkerboard Brightness Pattern",
        description="3x3 grid with alternating bright/dim spots - phase compensates for non-uniformity",
        generator=gen_spots_checkerboard_brightness,
    ),
    SampleConfig(
        id="spots_center_bright",
        level=3,
        category="spot_arrays",
        name="Center Spot Brightening",
        description="3x3 grid, center spot brightness increases - ratio 1:1 to 5:1",
        generator=gen_spots_center_bright,
    ),
    # Off-Center and Asymmetric
    SampleConfig(
        id="spots_diagonal_line",
        level=3,
        category="spot_arrays",
        name="Diagonal Line of Spots",
        description="Five spots along diagonal - spacing increases from tight to wide",
        generator=gen_spots_diagonal_line,
    ),
    SampleConfig(
        id="spots_L_shape",
        level=3,
        category="spot_arrays",
        name="L-Shaped Spot Array",
        description="Six spots forming L shape - asymmetric arrangement rotates",
        generator=gen_spots_L_shape,
    ),
    SampleConfig(
        id="spots_circle_arrangement",
        level=3,
        category="spot_arrays",
        name="Circular Arrangement",
        description="Eight spots arranged in circle - radius sweeps from small to large",
        generator=gen_spots_circle_arrangement,
    ),
    SampleConfig(
        id="spots_asymmetric_cross",
        level=3,
        category="spot_arrays",
        name="Asymmetric Cross",
        description="Five spots in cross pattern - arms at different lengths extend outward",
        generator=gen_spots_asymmetric_cross,
    ),
    SampleConfig(
        id="spots_triangle_vertices",
        level=3,
        category="spot_arrays",
        name="Triangle Vertices",
        description="Three spots at triangle corners - triangle rotates and scales",
        generator=gen_spots_triangle_vertices,
    ),
    # Random and Irregular Positions
    SampleConfig(
        id="spots_random_4",
        level=3,
        category="spot_arrays",
        name="Four Random Spots",
        description="Four spots at random positions - phase appears noisy, no obvious structure",
        generator=gen_spots_random_4,
    ),
    SampleConfig(
        id="spots_random_6",
        level=3,
        category="spot_arrays",
        name="Six Random Spots",
        description="Six spots randomly distributed - more complex phase optimization",
        generator=gen_spots_random_6,
    ),
    SampleConfig(
        id="spots_random_to_grid",
        level=3,
        category="spot_arrays",
        name="Random to Grid Transition",
        description="Four spots morph from random positions to 2x2 grid - watch phase organize",
        generator=gen_spots_random_to_grid,
    ),
    # Spot Size and Density Variations
    SampleConfig(
        id="spots_varying_size_3x3",
        level=3,
        category="spot_arrays",
        name="3x3 Varying Spot Sizes",
        description="Nine spots with different radii - center largest, corners smallest",
        generator=gen_spots_varying_size_3x3,
    ),
    SampleConfig(
        id="spots_dense_cluster",
        level=3,
        category="spot_arrays",
        name="Dense Spot Cluster",
        description="Nine spots tightly clustered in center - tests GS resolution limits",
        generator=gen_spots_dense_cluster,
    ),
]


# =============================================================================
# Level 4: Special Beams - Frame Generators
# =============================================================================

def gen_vortex_charge_sweep(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex beam with topological charge sweeping from 1 to 5."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        charge = 1 + int(4 * i / (n_frames - 1))  # 1, 2, 3, 4, 5
        phase = create_vortex_phase(GRID_SIZE, charge)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_charge_1_rotation(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex charge l=1 with phase offset rotation."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase = create_vortex_phase(GRID_SIZE, charge=1, phase_offset=phase_offset)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_charge_2(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex charge l=2 with rotation animation."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase = create_vortex_phase(GRID_SIZE, charge=2, phase_offset=phase_offset)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_charge_3(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex charge l=3 with rotation animation."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase = create_vortex_phase(GRID_SIZE, charge=3, phase_offset=phase_offset)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_opposite_charges(input_amp: np.ndarray, zoom: float = 1.0):
    """Two vortices with opposite charges at different positions."""
    frames = []
    n_frames = 32  # Extended from 16 to 32
    for i in range(n_frames):
        separation = 20 + 40 * i / (n_frames - 1)  # 20 to 60 pixels
        phase1 = create_vortex_phase(GRID_SIZE, charge=2, cx=-separation/2, cy=0)
        phase2 = create_vortex_phase(GRID_SIZE, charge=-2, cx=separation/2, cy=0)
        phase = np.mod(phase1 + phase2 + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_nested_vortex(input_amp: np.ndarray, zoom: float = 1.0):
    """Nested vortex structure with radius sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        radius = 20 + 40 * i / (n_frames - 1)  # 20 to 60 pixels
        x = np.arange(GRID_SIZE) - GRID_SIZE // 2
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)

        # Inner vortex l=1, outer vortex l=2
        phase_inner = create_vortex_phase(GRID_SIZE, charge=1)
        phase_outer = create_vortex_phase(GRID_SIZE, charge=2)
        phase = np.where(r < radius, phase_inner, phase_outer)

        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_axicon_slope_sweep(input_amp: np.ndarray, zoom: float = 1.0):
    """Axicon with conical slope increasing."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        slope = 0.01 + 0.09 * i / (n_frames - 1)  # 0.01 to 0.1
        phase = create_axicon_phase(GRID_SIZE, slope)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_bessel_order_0(input_amp: np.ndarray, zoom: float = 1.0):
    """Zero-order Bessel beam with slope variation."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        slope = 0.03 + 0.04 * i / (n_frames - 1)  # 0.03 to 0.07
        phase = create_axicon_phase(GRID_SIZE, slope=slope)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_bessel_order_1(input_amp: np.ndarray, zoom: float = 1.0):
    """First-order Bessel beam with rotation animation."""
    frames = []
    n_frames = 24
    phase_axicon = create_axicon_phase(GRID_SIZE, slope=0.05)
    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase_vortex = create_vortex_phase(GRID_SIZE, charge=1, phase_offset=phase_offset)
        phase = np.mod(phase_axicon + phase_vortex + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_axicon_truncated(input_amp: np.ndarray, zoom: float = 1.0):
    """Axicon with aperture size sweep."""
    frames = []
    n_frames = 32  # Extended from 16 to 32
    for i in range(n_frames):
        aperture_radius = 40 + 60 * i / (n_frames - 1)  # 40 to 100 pixels
        phase = create_axicon_phase(GRID_SIZE, slope=0.05)

        # Apply circular aperture
        x = np.arange(GRID_SIZE) - GRID_SIZE // 2
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X**2 + Y**2)
        phase = np.where(r <= aperture_radius, phase, 0)

        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_lg_radial_sweep(input_amp: np.ndarray, zoom: float = 1.0):
    """Laguerre-Gaussian with radial index p sweeping 0 to 3."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        p = int(3 * i / (n_frames - 1))  # 0, 1, 2, 3
        phase = create_laguerre_gaussian_phase(GRID_SIZE, p=p, l=0)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_lg_azimuthal_sweep(input_amp: np.ndarray, zoom: float = 1.0):
    """Laguerre-Gaussian with azimuthal index l sweeping 0 to 4."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        l = int(4 * i / (n_frames - 1))  # 0, 1, 2, 3, 4
        phase = create_laguerre_gaussian_phase(GRID_SIZE, p=0, l=l)
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_lg_p1_l1(input_amp: np.ndarray, zoom: float = 1.0):
    """Laguerre-Gaussian LG(1,1) mode with rotation animation."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase = create_laguerre_gaussian_phase(GRID_SIZE, p=1, l=1)
        # Add rotation by offsetting the vortex component
        x = np.arange(GRID_SIZE) - GRID_SIZE // 2
        X, Y = np.meshgrid(x, x)
        theta = np.arctan2(Y, X)
        phase = phase + 1 * phase_offset  # Rotate the l=1 component
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_lg_p2_l2(input_amp: np.ndarray, zoom: float = 1.0):
    """Laguerre-Gaussian LG(2,2) mode with rotation animation."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase = create_laguerre_gaussian_phase(GRID_SIZE, p=2, l=2)
        # Add rotation by offsetting the vortex component
        phase = phase + 2 * phase_offset  # Rotate the l=2 component
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_plus_lens(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex l=2 plus quadratic lens with curvature sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        curvature = 0.5 + 2.5 * i / (n_frames - 1)  # 0.5 to 3.0
        phase_vortex = create_vortex_phase(GRID_SIZE, charge=2)
        phase_lens = create_quadratic_phase(GRID_SIZE, curvature)
        phase = np.mod(phase_vortex + phase_lens + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_plus_grating(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex l=1 plus binary grating with period sweep."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        period = 64 - 48 * i / (n_frames - 1)  # 64 to 16 pixels
        phase_vortex = create_vortex_phase(GRID_SIZE, charge=1)
        phase_grating = create_binary_grating(GRID_SIZE, period=period, angle=0)
        phase = np.mod(phase_vortex + phase_grating + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_plus_tilt(input_amp: np.ndarray, zoom: float = 1.0):
    """Vortex l=2 plus linear tilt with angle sweep."""
    frames = []
    n_frames = 24
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        kx = 0.05 * np.cos(angle)
        ky = 0.05 * np.sin(angle)
        phase_vortex = create_vortex_phase(GRID_SIZE, charge=2)
        phase_tilt = create_linear_ramp(GRID_SIZE, kx=kx, ky=ky)
        phase = np.mod(phase_vortex + phase_tilt + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_double_vortex_separation(input_amp: np.ndarray, zoom: float = 1.0):
    """Two vortices l=1 with increasing separation."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        separation = 10 + 60 * i / (n_frames - 1)  # 10 to 70 pixels
        phase1 = create_vortex_phase(GRID_SIZE, charge=1, cx=-separation/2, cy=0)
        phase2 = create_vortex_phase(GRID_SIZE, charge=1, cx=separation/2, cy=0)
        phase = np.mod(phase1 + phase2 + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_vortex_array_2x2(input_amp: np.ndarray, zoom: float = 1.0):
    """2x2 array of vortices with synchronized rotation."""
    frames = []
    n_frames = 24
    spacing = 40
    charges = [1, -1, -1, 1]  # Alternating pattern
    positions = [
        (-spacing/2, -spacing/2),
        (spacing/2, -spacing/2),
        (-spacing/2, spacing/2),
        (spacing/2, spacing/2),
    ]

    for i in range(n_frames):
        phase_offset = 2 * np.pi * i / n_frames
        phase = np.zeros((GRID_SIZE, GRID_SIZE))
        for (cx, cy), charge in zip(positions, charges):
            phase += create_vortex_phase(GRID_SIZE, charge=charge, cx=cx, cy=cy, phase_offset=phase_offset)

        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


def gen_axicon_plus_vortex(input_amp: np.ndarray, zoom: float = 1.0):
    """Axicon plus vortex with charge sweep (high-order Bessel beams)."""
    frames = []
    n_frames = 16
    for i in range(n_frames):
        charge = int(3 * i / (n_frames - 1))  # 0, 1, 2, 3
        phase_axicon = create_axicon_phase(GRID_SIZE, slope=0.05)
        phase_vortex = create_vortex_phase(GRID_SIZE, charge=charge)
        phase = np.mod(phase_axicon + phase_vortex + np.pi, 2 * np.pi) - np.pi
        intensity = compute_intensity(input_amp, phase, zoom=zoom)
        frames.append((phase, intensity))
    return frames


# =============================================================================
# Level 4 Sample Configurations
# =============================================================================

L4_SAMPLES = [
    # Vortex Beams
    SampleConfig(
        id="vortex_charge_sweep",
        level=4,
        category="special_beams",
        name="Vortex Charge Sweep",
        description="Optical vortex with topological charge l=1→5 - donut radius increases with charge",
        generator=gen_vortex_charge_sweep,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="vortex_charge_1_rotation",
        level=4,
        category="special_beams",
        name="Vortex Charge 1 Rotation",
        description="Vortex l=1 with phase offset rotation - shows phase singularity at center",
        generator=gen_vortex_charge_1_rotation,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="vortex_charge_2",
        level=4,
        category="special_beams",
        name="Vortex Charge 2",
        description="Vortex l=2 with rotation - higher-order orbital angular momentum",
        generator=gen_vortex_charge_2,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="vortex_charge_3",
        level=4,
        category="special_beams",
        name="Vortex Charge 3",
        description="Vortex l=3 with rotation - demonstrates spiral phase structure",
        generator=gen_vortex_charge_3,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="vortex_opposite_charges",
        level=4,
        category="special_beams",
        name="Opposite Charge Vortices",
        description="Two vortices l=+2 and l=-2 with separation sweep - OAM interference and cancellation",
        generator=gen_vortex_opposite_charges,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="nested_vortex",
        level=4,
        category="special_beams",
        name="Nested Vortex",
        description="Nested vortex structure (l=1 inside, l=2 outside) - radius sweep",
        generator=gen_nested_vortex,
        intensity_zoom=5.0,
    ),

    # Axicon/Bessel Beams
    SampleConfig(
        id="axicon_slope_sweep",
        level=4,
        category="special_beams",
        name="Axicon Slope Sweep",
        description="Conical phase with increasing slope - Bessel ring spacing decreases",
        generator=gen_axicon_slope_sweep,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="bessel_order_0",
        level=4,
        category="special_beams",
        name="Bessel Beam Order 0",
        description="Zero-order Bessel beam with slope variation - ring spacing changes",
        generator=gen_bessel_order_0,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="bessel_order_1",
        level=4,
        category="special_beams",
        name="Bessel Beam Order 1",
        description="First-order Bessel beam with rotation - dark center with bright ring (axicon + vortex)",
        generator=gen_bessel_order_1,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="axicon_truncated",
        level=4,
        category="special_beams",
        name="Truncated Axicon",
        description="Axicon with aperture size sweep - extended animation showing transition to Bessel pattern",
        generator=gen_axicon_truncated,
        intensity_zoom=5.0,
    ),

    # Laguerre-Gaussian Beams
    SampleConfig(
        id="lg_radial_sweep",
        level=4,
        category="special_beams",
        name="LG Radial Index Sweep",
        description="Laguerre-Gaussian p=0→3, l=0 - radial ring count increases",
        generator=gen_lg_radial_sweep,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="lg_azimuthal_sweep",
        level=4,
        category="special_beams",
        name="LG Azimuthal Index Sweep",
        description="Laguerre-Gaussian l=0→4, p=0 - Gaussian to vortex transition",
        generator=gen_lg_azimuthal_sweep,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="lg_p1_l1",
        level=4,
        category="special_beams",
        name="LG(1,1) Mode",
        description="Laguerre-Gaussian mode with p=1, l=1 and rotation - radial and azimuthal structure",
        generator=gen_lg_p1_l1,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="lg_p2_l2",
        level=4,
        category="special_beams",
        name="LG(2,2) Mode",
        description="Laguerre-Gaussian mode with p=2, l=2 and rotation - higher-order mode",
        generator=gen_lg_p2_l2,
        intensity_zoom=5.0,
    ),

    # Compound Beams
    SampleConfig(
        id="vortex_plus_lens",
        level=4,
        category="special_beams",
        name="Vortex + Lens",
        description="Vortex l=2 plus quadratic lens - focused vortex beam, curvature sweep",
        generator=gen_vortex_plus_lens,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="vortex_plus_grating",
        level=4,
        category="special_beams",
        name="Vortex + Grating",
        description="Vortex l=1 plus binary grating - OAM transferred to diffraction orders",
        generator=gen_vortex_plus_grating,
        intensity_zoom=3.0,
    ),
    SampleConfig(
        id="vortex_plus_tilt",
        level=4,
        category="special_beams",
        name="Vortex + Tilt",
        description="Vortex l=2 plus linear tilt - off-axis vortex beam, angle sweep",
        generator=gen_vortex_plus_tilt,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="double_vortex_separation",
        level=4,
        category="special_beams",
        name="Double Vortex Separation",
        description="Two l=1 vortices with increasing separation - vortex interaction",
        generator=gen_double_vortex_separation,
        intensity_zoom=5.0,
    ),
    SampleConfig(
        id="vortex_array_2x2",
        level=4,
        category="special_beams",
        name="Vortex Array 2x2",
        description="2×2 array of vortices with alternating charges ±1 and synchronized rotation - complex OAM field",
        generator=gen_vortex_array_2x2,
        intensity_zoom=3.0,
    ),
    SampleConfig(
        id="axicon_plus_vortex",
        level=4,
        category="special_beams",
        name="Axicon + Vortex",
        description="Axicon plus vortex charge sweep l=0→3 - high-order Bessel beams",
        generator=gen_axicon_plus_vortex,
        intensity_zoom=5.0,
    ),
]


def get_all_samples() -> List[SampleConfig]:
    """Get all sample configurations."""
    return L1_SAMPLES + L2_SAMPLES + L3_SAMPLES + L4_SAMPLES


def generate_selected_samples(output_dir: Path, sample_ids: List[str]) -> dict:
    """
    Generate selected training samples by ID.

    Args:
        output_dir: Base output directory for assets
        sample_ids: List of sample IDs to generate

    Returns:
        Manifest dict with generated samples
    """
    input_amp = create_gaussian_input(GRID_SIZE)
    all_samples = get_all_samples()
    samples_by_id = {s.id: s for s in all_samples}

    # Validate IDs
    invalid_ids = [sid for sid in sample_ids if sid not in samples_by_id]
    if invalid_ids:
        print(f"Warning: Unknown sample IDs: {invalid_ids}")

    samples_to_gen = [samples_by_id[sid] for sid in sample_ids if sid in samples_by_id]
    manifest_entries = []

    print(f"Generating {len(samples_to_gen)} samples...")

    for i, config in enumerate(samples_to_gen):
        print(f"  [{i+1}/{len(samples_to_gen)}] {config.name}...")
        entry = generate_sample(config, output_dir, input_amp)
        manifest_entries.append(entry)

    manifest = {
        "samples": manifest_entries,
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }

    # Note: Don't overwrite full manifest when generating subset
    print(f"Generated {len(manifest_entries)} samples (manifest not updated)")
    return manifest


def generate_all_samples(output_dir: Path) -> dict:
    """
    Generate all training samples.

    Args:
        output_dir: Base output directory for assets

    Returns:
        Complete manifest dict
    """
    input_amp = create_gaussian_input(GRID_SIZE)
    samples = get_all_samples()
    manifest_entries = []

    print(f"Generating {len(samples)} samples...")

    for i, config in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {config.name}...")
        entry = generate_sample(config, output_dir, input_amp)
        manifest_entries.append(entry)

    manifest = {
        "samples": manifest_entries,
        "generated_at": datetime.now().isoformat(),
        "version": "1.0.0",
    }

    # Save manifest
    manifest_path = output_dir.parent / "samples.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    return manifest
