"""
Pattern Generators for Level 4: Special Beams

Vortex, Bessel, Laguerre-Gaussian, and compound beam patterns.
"""

import numpy as np
from scipy.special import genlaguerre, jv
from typing import Tuple


def create_vortex_phase(
    size: int, charge: int, cx: float = 0, cy: float = 0, phase_offset: float = 0
) -> np.ndarray:
    """
    Create optical vortex phase pattern.
    
    Args:
        size: Grid size
        charge: Topological charge (integer, positive or negative)
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
        phase_offset: Additional phase rotation (radians)
    
    Returns:
        Phase mask with spiral structure
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    X = X - cx
    Y = Y - cy
    
    theta = np.arctan2(Y, X)
    phase = charge * theta + phase_offset
    
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_axicon_phase(
    size: int, slope: float, cx: float = 0, cy: float = 0
) -> np.ndarray:
    """
    Create axicon (conical lens) phase pattern.
    
    Args:
        size: Grid size
        slope: Conical slope (radians per pixel)
        cx: Center x position (pixels from center)
        cy: Center y position (pixels from center)
    
    Returns:
        Phase mask with conical structure
    """
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    X = X - cx
    Y = Y - cy
    
    r = np.sqrt(X**2 + Y**2)
    phase = slope * r
    
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_laguerre_gaussian_phase(
    size: int, p: int, l: int, w0: float = None
) -> np.ndarray:
    """
    Create phase pattern for generating LG-like beams via phase-only SLM.

    For phase-only SLMs, we can only impart the vortex (azimuthal) phase.
    The radial index p affects the radial structure but cannot be fully
    reproduced with phase-only modulation. We add concentric ring phase
    jumps to approximate the radial nodes.

    Args:
        size: Grid size
        p: Radial index (number of radial nodes) - approximated with ring phase jumps
        l: Azimuthal index (OAM charge) - exact vortex phase
        w0: Beam waist radius (defaults to size/4)

    Returns:
        Phase mask for LG-like mode
    """
    if w0 is None:
        w0 = size / 4

    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # Vortex phase for azimuthal index l
    vortex_phase = l * theta

    # For radial index p > 0, add concentric ring phase jumps (pi phase shifts)
    # This approximates the sign changes in the Laguerre polynomial
    radial_phase = np.zeros_like(r)
    if p > 0:
        # Add p concentric rings with pi phase jumps
        for ring in range(1, p + 1):
            ring_radius = w0 * np.sqrt(2 * ring)  # Approximate node positions
            radial_phase += np.pi * (r > ring_radius).astype(float)

    phase = vortex_phase + radial_phase

    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def create_laguerre_gaussian_target(
    size: int, p: int, l: int, w0: float = None
) -> np.ndarray:
    """
    Create Laguerre-Gaussian mode intensity pattern.
    
    Args:
        size: Grid size
        p: Radial index (number of radial nodes)
        l: Azimuthal index (OAM charge)
        w0: Beam waist radius (defaults to size/4)
    
    Returns:
        Target amplitude for LG mode
    """
    if w0 is None:
        w0 = size / 4
    
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    
    r = np.sqrt(X**2 + Y**2)
    
    # Normalized radial coordinate
    rho = np.sqrt(2) * r / w0
    
    # Laguerre polynomial
    abs_l = abs(l)
    laguerre = genlaguerre(p, abs_l)
    
    # LG amplitude
    amplitude = (rho ** abs_l) * laguerre(rho**2) * np.exp(-rho**2 / 2)
    
    # Normalize
    if amplitude.max() > 0:
        amplitude = amplitude / amplitude.max()
    
    return amplitude


def create_bessel_target(
    size: int, order: int, kr: float, radius: float = None
) -> np.ndarray:
    """
    Create Bessel beam intensity pattern.
    
    Args:
        size: Grid size
        order: Bessel function order (0, 1, 2, ...)
        kr: Radial wave vector (controls ring spacing)
        radius: Maximum radius to compute (defaults to size/2)
    
    Returns:
        Target amplitude for Bessel beam
    """
    if radius is None:
        radius = size / 2
    
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    
    # Bessel function J_n(kr*r)
    amplitude = np.abs(jv(order, kr * r))
    
    # Apply aperture
    amplitude = np.where(r <= radius, amplitude, 0)
    
    # Normalize
    if amplitude.max() > 0:
        amplitude = amplitude / amplitude.max()
    
    return amplitude

