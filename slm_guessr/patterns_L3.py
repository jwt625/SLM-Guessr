"""
Pattern Generators for Level 3: Discrete Spot Arrays

Multi-spot target intensity generators for GS-optimized phase masks.
"""

import numpy as np
from typing import List, Tuple


def create_grid_spots(
    size: int,
    rows: int,
    cols: int,
    spacing_x: float,
    spacing_y: float = None,
    radius: int = 4,
    brightness: np.ndarray = None,
) -> np.ndarray:
    """
    Create regular grid of spots.
    
    Args:
        size: Grid size
        rows: Number of rows
        cols: Number of columns
        spacing_x: Horizontal spacing between spots (pixels)
        spacing_y: Vertical spacing (defaults to spacing_x if None)
        radius: Spot radius in pixels
        brightness: Optional brightness array (rows × cols), normalized to max=1
    
    Returns:
        Target amplitude
    """
    if spacing_y is None:
        spacing_y = spacing_x
    
    if brightness is None:
        brightness = np.ones((rows, cols))
    else:
        brightness = brightness / brightness.max()
    
    target = np.zeros((size, size))
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    
    # Center the grid
    total_width = (cols - 1) * spacing_x
    total_height = (rows - 1) * spacing_y
    start_x = -total_width / 2
    start_y = -total_height / 2
    
    for i in range(rows):
        for j in range(cols):
            cx = start_x + j * spacing_x
            cy = start_y + i * spacing_y
            r2 = (X - cx)**2 + (Y - cy)**2
            mask = r2 <= radius**2
            target[mask] = np.maximum(target[mask], brightness[i, j])
    
    return target


def create_spots_at_positions(
    size: int,
    positions: List[Tuple[float, float]],
    radius: int = 4,
    brightness: List[float] = None,
) -> np.ndarray:
    """
    Create spots at arbitrary positions.
    
    Args:
        size: Grid size
        positions: List of (cx, cy) positions in pixels from center
        radius: Spot radius in pixels
        brightness: Optional brightness for each spot (normalized to max=1)
    
    Returns:
        Target amplitude
    """
    if brightness is None:
        brightness = [1.0] * len(positions)
    else:
        max_b = max(brightness)
        brightness = [b / max_b for b in brightness]
    
    target = np.zeros((size, size))
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    
    for (cx, cy), b in zip(positions, brightness):
        r2 = (X - cx)**2 + (Y - cy)**2
        mask = r2 <= radius**2
        target[mask] = np.maximum(target[mask], b)
    
    return target


def create_random_spot_positions(
    n_spots: int,
    size: int,
    min_separation: float = 15.0,
    max_radius: float = None,
    seed: int = None,
) -> List[Tuple[float, float]]:
    """
    Generate random spot positions with minimum separation constraint.
    
    Args:
        n_spots: Number of spots
        size: Grid size
        min_separation: Minimum distance between spot centers
        max_radius: Maximum distance from center (defaults to size/3)
        seed: Random seed for reproducibility
    
    Returns:
        List of (cx, cy) positions
    """
    if max_radius is None:
        max_radius = size / 3
    
    if seed is not None:
        np.random.seed(seed)
    
    positions = []
    max_attempts = 1000
    
    for _ in range(n_spots):
        for attempt in range(max_attempts):
            # Random position within max_radius
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, max_radius)
            cx = r * np.cos(angle)
            cy = r * np.sin(angle)
            
            # Check minimum separation
            valid = True
            for (px, py) in positions:
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < min_separation:
                    valid = False
                    break
            
            if valid:
                positions.append((cx, cy))
                break
    
    return positions

