<p align="center">
  <img src="static/icon.svg" alt="SLM-Guessr" width="128" height="128" />
</p>

# SLM-Guessr

An educational web application for training intuition on Spatial Light Modulator (SLM) phase masks and their corresponding 2D FFT intensity patterns.

## Overview

SLM-Guessr presents animated phase-intensity pairs progressing from simple to complex patterns, enabling users to build pattern recognition skills for holographic optics applications. The training curriculum spans seven levels covering foundations, periodic structures, spot arrays, special beams, compound patterns, practical applications, and complex shapes.

## Features

- **Gallery Mode**: Browse annotated phase-intensity pairs organized by level and category
- **Quiz Mode**: Test comprehension with bidirectional challenges (phase-to-intensity or intensity-to-phase)
- **Progressive Curriculum**: 150+ samples across 7 difficulty levels
- **Animated Demonstrations**: Parameter sweeps showing continuous phase-intensity relationships

## Curriculum

| Level | Topic | Sample Count |
|-------|-------|--------------|
| L1 | Foundations (uniform phase, linear ramps, quadratic phase) | 20 |
| L2 | Periodic Structures (gratings, checkerboards) | 20 |
| L3 | Discrete Spot Arrays | 20 |
| L4 | Special Beams (vortex, axicon, Laguerre-Gaussian) | 20 |
| L5 | Compound Patterns | 15 |
| L6 | Practical Applications (optical tweezers, beam steering) | 15 |
| L7 | Complex Shapes (letters, symbols, objects) | 40 |

## Technology Stack

**Frontend**: SvelteKit with TypeScript, static site generation for GitHub Pages deployment

**Content Generation**: Python with NumPy, Matplotlib, SciPy, and ImageIO for phase mask computation and GIF rendering

**Algorithms**: Gerchberg-Saxton phase retrieval with weighted variants for complex target patterns

## Development

### Prerequisites

- Node.js 18+
- pnpm
- Python 3.10+ (for content generation)

### Setup

```bash
# Install frontend dependencies
pnpm install

# Create Python virtual environment
uv venv
source .venv/bin/activate
uv pip install numpy matplotlib scipy imageio

# Generate training samples
python slm_guessr/generate_samples.py
```

### Development Server

```bash
pnpm dev
```

### Build

```bash
pnpm build
```

Output is generated in `build/` directory for static deployment.

## Deployment

Configured for GitHub Pages deployment under `/SLM-Guessr` subpath. The base path is set in `svelte.config.js` and automatically applied in production builds.

## Project Structure

```
slm-guessr/
├── src/                    # SvelteKit application
│   ├── lib/
│   │   ├── components/     # GifPlayer, SampleCard, Header
│   │   └── types.ts        # TypeScript definitions
│   └── routes/             # Pages (home, gallery, quiz)
├── static/
│   ├── assets/             # Generated GIFs (L1-L7)
│   └── samples.json        # Sample manifest
├── slm_guessr/             # Python content generators
│   ├── patterns.py         # Pattern generation functions
│   └── generate_samples.py # Batch generation script
└── DevLog/                 # Development documentation
```

## References

- Gerchberg, R. W., & Saxton, W. O. (1972). A practical algorithm for the determination of phase from image and diffraction plane pictures. Optik, 35, 237-246.
- Di Leonardo, R., Ianni, F., & Ruocco, G. (2007). Computer generation of optimal holograms for optical trap arrays. Optics Express, 15(4), 1913-1922.
- Goodman, J. W. Introduction to Fourier Optics.

## License

MIT

