"""
Pattern Generators for Level 2: Periodic Structures

Advanced periodic pattern generators for SLM-Guessr training samples.
"""

import numpy as np
from typing import Tuple


def create_concentric_rings_binary(
    size: int, period: float, duty_cycle: float = 0.5
) -> np.ndarray:
    """
    Create concentric binary rings pattern.
    
    Args:
        size: Grid size
        period: Radial period in pixels
        duty_cycle: Fraction of period that is high phase (0 to 1)
    
    Returns:
        Phase mask (0 or pi)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    phase_frac = np.mod(R / period, 1.0)
    phase = np.where(phase_frac < duty_cycle, np.pi, 0.0)
    return phase


def create_concentric_rings_sinusoidal(
    size: int, period: float, amplitude: float = np.pi
) -> np.ndarray:
    """
    Create concentric sinusoidal rings (Fresnel zone plate-like).
    
    Args:
        size: Grid size
        period: Radial period in pixels
        amplitude: Phase modulation amplitude
    
    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    phase = amplitude * np.sin(2 * np.pi * R / period)
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_radial_sectors(size: int, n_sectors: int) -> np.ndarray:
    """
    Create radial sectors pattern (pizza slices).
    
    Args:
        size: Grid size
        n_sectors: Number of sectors
    
    Returns:
        Phase mask (0 or pi alternating)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    theta = np.arctan2(Y, X)  # -pi to pi
    # Normalize to 0 to 1
    theta_norm = (theta + np.pi) / (2 * np.pi)
    sector_index = np.floor(theta_norm * n_sectors).astype(int)
    phase = np.where(sector_index % 2 == 0, 0.0, np.pi)
    return phase


def create_spiral_grating(
    size: int, pitch: float, n_arms: int = 1
) -> np.ndarray:
    """
    Create Archimedean spiral grating.
    
    Args:
        size: Grid size
        pitch: Radial pitch (spacing between spiral arms)
        n_arms: Number of spiral arms
    
    Returns:
        Phase mask (0 or pi)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    # Archimedean spiral: r = a * theta
    # For binary pattern: check if (R - a*theta) mod period < duty_cycle
    spiral_phase = R - pitch * theta * n_arms / (2 * np.pi)
    phase_frac = np.mod(spiral_phase / pitch, 1.0)
    phase = np.where(phase_frac < 0.5, 0.0, np.pi)
    return phase


def create_hexagonal_lattice(size: int, period: float) -> np.ndarray:
    """
    Create hexagonal lattice pattern.
    
    Args:
        size: Grid size
        period: Lattice period in pixels
    
    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    
    # Hexagonal lattice using three gratings at 60° angles
    k = 2 * np.pi / period
    g1 = np.cos(k * X)
    g2 = np.cos(k * (X * np.cos(np.pi/3) + Y * np.sin(np.pi/3)))
    g3 = np.cos(k * (X * np.cos(2*np.pi/3) + Y * np.sin(2*np.pi/3)))
    
    # Combine and threshold
    combined = g1 + g2 + g3
    phase = np.where(combined > 0, np.pi, 0.0)
    return phase


def create_triangular_lattice(size: int, period: float) -> np.ndarray:
    """
    Create triangular tiling pattern (filled triangles).

    Args:
        size: Grid size
        period: Lattice period in pixels

    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    # Create triangular tiling by using three gratings
    # Each grating divides space into stripes at 120° angles
    k = 2 * np.pi / (period * np.sqrt(3))  # Adjust for triangular geometry

    # Three sets of parallel lines at 120° angles
    g1 = np.floor(np.mod(k * Y / (2*np.pi), 1) * 2)  # Horizontal stripes
    g2 = np.floor(np.mod(k * (Y * np.cos(np.pi/3) - X * np.sin(np.pi/3)) / (2*np.pi), 1) * 2)
    g3 = np.floor(np.mod(k * (Y * np.cos(np.pi/3) + X * np.sin(np.pi/3)) / (2*np.pi), 1) * 2)

    # XOR combination creates triangular tiling
    combined = (g1.astype(int) + g2.astype(int) + g3.astype(int)) % 2
    phase = np.where(combined == 0, 0.0, np.pi)

    return phase


def create_grating_with_defect(
    size: int, period: float, defect_position: float, defect_width: float = 10
) -> np.ndarray:
    """
    Create binary grating with a phase defect (pi phase shift).

    Args:
        size: Grid size
        period: Grating period in pixels
        defect_position: X position of defect center (-size/2 to size/2)
        defect_width: Width of defect region in pixels

    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    # Base grating
    phase_frac = np.mod(X / period, 1.0)
    phase = np.where(phase_frac < 0.5, np.pi, 0.0)

    # Add defect (pi phase shift in a region)
    defect_mask = np.abs(X - defect_position) < defect_width / 2
    phase = np.where(defect_mask, np.mod(phase + np.pi, 2*np.pi), phase)

    return phase


def create_chirped_grating(
    size: int, period_start: float, period_end: float
) -> np.ndarray:
    """
    Create chirped grating (linearly varying period).

    Args:
        size: Grid size
        period_start: Period at left edge
        period_end: Period at right edge

    Returns:
        Phase mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    # Linear chirp: period varies with X
    # Integrate to get phase: phi = integral(2*pi/period(x) dx)
    x_norm = (X + size/2) / size  # 0 to 1
    period_x = period_start + (period_end - period_start) * x_norm

    # Approximate chirped phase
    k_avg = 2 * np.pi / ((period_start + period_end) / 2)
    chirp_rate = np.pi * (period_end - period_start) / (size * period_start * period_end)
    phase_continuous = k_avg * X + chirp_rate * X**2

    # Binarize
    phase = np.where(np.mod(phase_continuous / np.pi, 2) < 1, 0.0, np.pi)
    return phase


def create_duty_cycle_grating(
    size: int, period: float, duty_cycle: float, angle: float = 0
) -> np.ndarray:
    """
    Create binary grating with specified duty cycle.

    Args:
        size: Grid size
        period: Grating period in pixels
        duty_cycle: Fraction of period that is high (0 to 1)
        angle: Rotation angle in radians

    Returns:
        Phase mask (0 or pi)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    # Rotate coordinates
    Xr = X * np.cos(angle) + Y * np.sin(angle)

    # Create grating with duty cycle
    phase_frac = np.mod(Xr / period, 1.0)
    phase = np.where(phase_frac < duty_cycle, np.pi, 0.0)
    return phase


def create_amplitude_modulated_grating(
    size: int, period: float, envelope_sigma: float = None
) -> np.ndarray:
    """
    Create grating with Gaussian amplitude envelope.

    Args:
        size: Grid size
        period: Grating period in pixels
        envelope_sigma: Gaussian envelope width (default: size/4)

    Returns:
        Phase mask (sinusoidal with Gaussian envelope)
    """
    if envelope_sigma is None:
        envelope_sigma = size / 4

    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    # Gaussian envelope
    envelope = np.exp(-(X**2 + Y**2) / (2 * envelope_sigma**2))

    # Sinusoidal grating
    grating = np.sin(2 * np.pi * X / period)

    # Modulate phase amplitude by envelope
    phase = np.pi * envelope * grating
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_blazed_grating_variable(
    size: int, period: float, blaze_fraction: float, angle: float = 0
) -> np.ndarray:
    """
    Create blazed grating with variable blaze angle.

    Args:
        size: Grid size
        period: Grating period in pixels
        blaze_fraction: Blaze slope as fraction of 2*pi (0 to 1)
        angle: Rotation angle in radians

    Returns:
        Phase mask (sawtooth)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)

    # Rotate coordinates
    Xr = X * np.cos(angle) + Y * np.sin(angle)

    # Sawtooth wave
    phase_frac = np.mod(Xr / period, 1.0)
    phase = 2 * np.pi * blaze_fraction * phase_frac
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_rings_and_sectors(
    size: int, ring_period: float, n_sectors: int
) -> np.ndarray:
    """
    Create combined concentric rings and radial sectors pattern.

    Args:
        size: Grid size
        ring_period: Radial period for rings in pixels
        n_sectors: Number of radial sectors

    Returns:
        Phase mask (XOR of rings and sectors)
    """
    x = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Concentric rings (binary)
    ring_frac = np.mod(R / ring_period, 1.0)
    rings = (ring_frac < 0.5).astype(int)

    # Radial sectors
    theta_norm = (theta + np.pi) / (2 * np.pi)  # 0 to 1
    sector_index = np.floor(theta_norm * n_sectors).astype(int)
    sectors = (sector_index % 2).astype(int)

    # XOR combination creates grid pattern in polar coordinates
    combined = (rings ^ sectors).astype(float)
    phase = np.where(combined == 1, np.pi, 0.0)

    return phase

