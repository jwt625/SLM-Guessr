"""
Pattern Generators for SLM-Guessr

Phase mask and target intensity pattern generators for training samples.
"""

import numpy as np
from typing import Tuple


def create_gaussian_input(size: int, sigma: float = None) -> np.ndarray:
    """Create Gaussian input beam amplitude."""
    if sigma is None:
        sigma = size / 4
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / (2 * sigma**2))


def create_uniform_phase(size: int) -> np.ndarray:
    """Create uniform (zero) phase mask."""
    return np.zeros((size, size))


def create_linear_ramp(
    size: int, kx: float = 0, ky: float = 0
) -> np.ndarray:
    """
    Create linear phase ramp.
    
    Args:
        size: Grid size
        kx: Spatial frequency in x (radians per pixel)
        ky: Spatial frequency in y (radians per pixel)
    
    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    phase = kx * X + ky * Y
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_quadratic_phase(size: int, curvature: float) -> np.ndarray:
    """
    Create quadratic (lens-like) phase.
    
    Args:
        size: Grid size
        curvature: Curvature coefficient (positive = converging, negative = diverging)
    
    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    r2 = X**2 + Y**2
    phase = curvature * r2 / (size**2) * 4 * np.pi
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_cubic_phase(
    size: int, coeff_x: float = 0, coeff_y: float = 0
) -> np.ndarray:
    """
    Create cubic phase pattern.
    
    Args:
        size: Grid size
        coeff_x: Cubic coefficient in x
        coeff_y: Cubic coefficient in y
    
    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    phase = coeff_x * X**3 + coeff_y * Y**3
    phase = phase * np.pi
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_spot_target(
    size: int, cx: float, cy: float, radius: int = 3
) -> np.ndarray:
    """
    Create single spot target intensity.
    
    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        radius: Spot radius in pixels
    
    Returns:
        Target amplitude (sqrt of intensity)
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r2 = (X - cx)**2 + (Y - cy)**2
    target = np.zeros((size, size))
    target[r2 <= radius**2] = 1.0
    return target


def create_gaussian_spot_target(
    size: int, cx: float, cy: float, sigma: float = 5.0
) -> np.ndarray:
    """
    Create Gaussian spot target (soft edges, no sinc ringing).
    
    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        sigma: Gaussian width in pixels
    
    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r2 = (X - cx)**2 + (Y - cy)**2
    return np.exp(-r2 / (2 * sigma**2))


def create_rectangular_slab_target(
    size: int, cx: float, cy: float, width: int = 20, height: int = 40
) -> np.ndarray:
    """
    Create rectangular slab target.

    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        width: Rectangle width in pixels
        height: Rectangle height in pixels

    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    target = np.zeros((size, size))
    mask = (np.abs(X - cx) <= width // 2) & (np.abs(Y - cy) <= height // 2)
    target[mask] = 1.0
    return target


def create_line_target(
    size: int, cx: float, cy: float, width: int, orientation: str = 'vertical'
) -> np.ndarray:
    """
    Create thin line target (vertical or horizontal).

    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        width: Line width in pixels
        orientation: 'vertical' or 'horizontal'

    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    target = np.zeros((size, size))

    if orientation == 'vertical':
        mask = np.abs(X - cx) <= width // 2
    else:  # horizontal
        mask = np.abs(Y - cy) <= width // 2

    target[mask] = 1.0
    return target


def create_band_target(
    size: int, cx: float, cy: float, width: int, orientation: str = 'vertical',
    profile: str = 'uniform'
) -> np.ndarray:
    """
    Create band target with various intensity profiles.

    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        width: Band width in pixels
        orientation: 'vertical' or 'horizontal'
        profile: 'uniform' or 'half_sine'

    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    target = np.zeros((size, size))

    if orientation == 'vertical':
        dist = np.abs(X - cx)
    else:  # horizontal
        dist = np.abs(Y - cy)

    mask = dist <= width // 2

    if profile == 'uniform':
        target[mask] = 1.0
    elif profile == 'half_sine':
        # Half-sine profile across the width
        normalized_dist = dist / (width / 2)
        target[mask] = np.cos(normalized_dist[mask] * np.pi / 2)

    return target


def create_ellipse_target(
    size: int, cx: float, cy: float, radius_x: float, radius_y: float, angle: float = 0
) -> np.ndarray:
    """
    Create elliptical target.

    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        radius_x: Semi-major axis in x
        radius_y: Semi-minor axis in y
        angle: Rotation angle in radians

    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)

    # Translate to center
    Xt = X - cx
    Yt = Y - cy

    # Rotate
    Xr = Xt * np.cos(angle) + Yt * np.sin(angle)
    Yr = -Xt * np.sin(angle) + Yt * np.cos(angle)

    # Ellipse equation
    ellipse_eq = (Xr / radius_x)**2 + (Yr / radius_y)**2

    target = np.zeros((size, size))
    target[ellipse_eq <= 1.0] = 1.0
    return target


def create_ring_target(
    size: int, cx: float, cy: float, inner_radius: float, outer_radius: float
) -> np.ndarray:
    """
    Create ring (annulus) target.

    Args:
        size: Grid size
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        inner_radius: Inner radius in pixels
        outer_radius: Outer radius in pixels

    Returns:
        Target amplitude
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)

    target = np.zeros((size, size))
    mask = (r >= inner_radius) & (r <= outer_radius)
    target[mask] = 1.0
    return target


def create_ring_aperture_phase(
    size: int, inner_radius: float, outer_radius: float
) -> np.ndarray:
    """
    Create ring aperture (annular mask) - returns phase for use with masked input.
    This is used by multiplying input amplitude with the aperture mask.

    Args:
        size: Grid size
        inner_radius: Inner radius in pixels
        outer_radius: Outer radius in pixels

    Returns:
        Aperture mask (0 or 1)
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)

    mask = np.zeros((size, size))
    annulus = (r >= inner_radius) & (r <= outer_radius)
    mask[annulus] = 1.0
    return mask


def create_cylindrical_lens_phase(
    size: int, curvature: float, axis: str = 'x'
) -> np.ndarray:
    """
    Create cylindrical lens phase (quadratic in one direction only).

    Args:
        size: Grid size
        curvature: Curvature coefficient
        axis: 'x' or 'y' - direction of curvature

    Returns:
        Phase mask wrapped to [-pi, pi]
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    if axis == 'x':
        phase = curvature * X**2 / (size**2) * 4 * np.pi
    else:  # y
        phase = curvature * Y**2 / (size**2) * 4 * np.pi

    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_two_spots_target(
    size: int, separation: float, angle: float = 0
) -> np.ndarray:
    """
    Create two spots target.

    Args:
        size: Grid size
        separation: Distance between spots (pixels)
        angle: Angle of separation line (radians, 0 = horizontal)

    Returns:
        Target amplitude
    """
    half_sep = separation / 2
    cx1 = half_sep * np.cos(angle)
    cy1 = half_sep * np.sin(angle)
    cx2 = -half_sep * np.cos(angle)
    cy2 = -half_sep * np.sin(angle)

    spot1 = create_spot_target(size, cx1, cy1, radius=4)
    spot2 = create_spot_target(size, cx2, cy2, radius=4)

    return np.maximum(spot1, spot2)


# =============================================================================
# Level 2: Periodic Structures - Phase Patterns
# =============================================================================

def create_binary_grating(
    size: int, period: float, angle: float = 0, duty_cycle: float = 0.5
) -> np.ndarray:
    """
    Create binary phase grating.

    Args:
        size: Grid size
        period: Grating period in pixels
        angle: Rotation angle in radians
        duty_cycle: Fraction of period that is high (0 to 1)

    Returns:
        Phase mask (0 or pi)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    # Rotate coordinates
    Xr = X * np.cos(angle) + Y * np.sin(angle)
    # Create grating
    phase_frac = np.mod(Xr / period, 1.0)
    phase = np.where(phase_frac < duty_cycle, np.pi, 0.0)
    return phase


def create_sinusoidal_grating(
    size: int, period: float, angle: float = 0, amplitude: float = np.pi
) -> np.ndarray:
    """
    Create sinusoidal phase grating.

    Args:
        size: Grid size
        period: Grating period in pixels
        angle: Rotation angle in radians
        amplitude: Phase modulation amplitude

    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    Xr = X * np.cos(angle) + Y * np.sin(angle)
    phase = amplitude * np.sin(2 * np.pi * Xr / period)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_blazed_grating(size: int, period: float, angle: float = 0) -> np.ndarray:
    """
    Create blazed (sawtooth) phase grating.

    Args:
        size: Grid size
        period: Grating period in pixels
        angle: Rotation angle in radians

    Returns:
        Phase mask (sawtooth from -pi to pi)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    Xr = X * np.cos(angle) + Y * np.sin(angle)
    phase = np.mod(Xr / period, 1.0) * 2 * np.pi - np.pi
    return phase


def create_checkerboard(size: int, period: float) -> np.ndarray:
    """
    Create checkerboard phase pattern.

    Args:
        size: Grid size
        period: Checker size in pixels

    Returns:
        Phase mask (0 or pi in checkerboard pattern)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    checker_x = np.floor(X / period) % 2
    checker_y = np.floor(Y / period) % 2
    phase = np.where((checker_x + checker_y) % 2 == 0, 0.0, np.pi)
    return phase


def create_crossed_gratings(
    size: int, period: float, angle: float = 0
) -> np.ndarray:
    """
    Create crossed binary gratings (two orthogonal gratings).

    Args:
        size: Grid size
        period: Grating period in pixels
        angle: Rotation of the crossed pair in radians

    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    # First grating (along rotated X)
    Xr = X * np.cos(angle) + Y * np.sin(angle)
    Yr = -X * np.sin(angle) + Y * np.cos(angle)
    grating1 = np.mod(Xr / period, 1.0) < 0.5
    grating2 = np.mod(Yr / period, 1.0) < 0.5
    # XOR the two gratings
    phase = np.where(grating1 ^ grating2, np.pi, 0.0)
    return phase


def create_multi_frequency_grating(
    size: int, period1: float, period2: float, angle: float = 0
) -> np.ndarray:
    """
    Create grating with two frequency components.

    Args:
        size: Grid size
        period1: First grating period in pixels
        period2: Second grating period in pixels
        angle: Rotation angle in radians

    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    Xr = X * np.cos(angle) + Y * np.sin(angle)
    # Superposition of two sinusoidal gratings
    phase = np.pi * (np.sin(2 * np.pi * Xr / period1) +
                     0.5 * np.sin(2 * np.pi * Xr / period2))
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def compute_intensity(
    input_amplitude: np.ndarray, phase_mask: np.ndarray, zoom: float = 1.0
) -> np.ndarray:
    """
    Compute Fourier plane intensity from phase mask.

    Args:
        input_amplitude: Input beam amplitude
        phase_mask: Phase mask in radians
        zoom: Zoom factor for Fourier plane (>1 zooms in, showing smaller region with higher resolution)

    Returns:
        Intensity at Fourier plane
    """
    field = input_amplitude * np.exp(1j * phase_mask)

    if zoom > 1.0:
        # Zero-pad the field to increase Fourier plane resolution
        # Padding by zoom factor gives zoom times finer sampling in Fourier space
        size = field.shape[0]
        padded_size = int(size * zoom)
        pad_width = (padded_size - size) // 2

        field_padded = np.pad(field, ((pad_width, pad_width), (pad_width, pad_width)),
                             mode='constant', constant_values=0)

        # Compute FFT on padded field
        fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_padded)))

        # Extract central region (same size as original)
        center = padded_size // 2
        half_size = size // 2
        fourier_field = fourier_field[center - half_size:center + half_size,
                                     center - half_size:center + half_size]
    else:
        fourier_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

    return np.abs(fourier_field) ** 2

