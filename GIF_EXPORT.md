# GIF Export Guide for EFI

The EFI interactive viewer now supports exporting episodes as animated GIFs for easy sharing!

## Quick Start

### Method 1: Standalone GIF Exporter (Recommended)

Use the dedicated `export_gif.py` script to create GIFs from any episode:

```bash
# Export both full and simple GIFs
python export_gif.py --seed 6 --H 20 --W 20 --max-steps 200 --nA 3 --nB 3

# Export only simple GIF (smaller file size, great for sharing)
python export_gif.py --seed 6 --mode simple

# Export only full GIF (shows all CA fields)
python export_gif.py --seed 6 --mode full
```

### Method 2: Interactive Viewer (Requires Display)

If you have a display environment (local machine or X11 forwarding):

1. Run the interactive viewer:
```bash
python cli.py interactive --seed 6
```

2. Click the **"Export GIF"** button for full multi-panel export
3. Click the **"Simple GIF"** button for world-only export

## Export Options

### Full GIF
- **Size**: ~1MB for 200 frames
- **Content**: Shows all 6 field visualizations (World, GA/GB scents, Potential, Trail, Info)
- **Use Case**: Technical documentation, detailed analysis, presentations
- **Example filename**: `efi_full_20250817_162646.gif`

### Simple GIF  
- **Size**: ~100KB for 200 frames (90% smaller!)
- **Content**: Shows only the world view with agent, targets, and score
- **Use Case**: Social media (Twitter/X, Reddit, Discord), quick demos
- **Example filename**: `efi_simple_20250817_162646.gif`

## Parameters

### Episode Configuration
- `--seed`: Random seed for reproducibility
- `--H`, `--W`: Grid dimensions (default: 20x20)
- `--max-steps`: Episode length (default: 200)
- `--nA`, `--nB`: Number of A and B targets
- `--p-wall`: Wall generation probability

### Export Settings
- `--mode`: `full`, `simple`, or `both` (default: both)
- `--fps`: Frames per second (default: 8)
- `--output-dir`: Output directory (default: exports/)

## Examples

### Small, fast episode for testing
```bash
python export_gif.py --seed 42 --H 15 --W 15 --max-steps 100 --mode simple
```

### Large, complex environment
```bash
python export_gif.py --seed 6 --H 40 --W 40 --max-steps 500 --nA 20 --nB 30 --mode simple
```

### Specific scenario
```bash
# Dense environment with many targets
python export_gif.py --seed 123 --H 30 --W 30 --nA 10 --nB 15 --p-wall 0.15
```

## Tips

1. **For social media**: Use `--mode simple` for smaller files that load quickly
2. **For analysis**: Use `--mode full` to see all field dynamics
3. **Large grids**: Episodes with H,W > 30 may take 30+ seconds to generate
4. **File location**: GIFs are saved to `exports/` directory with timestamps

## Headless/SSH Usage

The standalone exporter works perfectly on headless servers:

```bash
# SSH into server
ssh your-server

# Run episode and export
python export_gif.py --seed 6 --mode simple

# Download the GIF
scp your-server:path/to/exports/efi_simple_*.gif ./
```

## HTML Viewer Export

When using the HTML viewer (in headless mode), click the export buttons for instructions on using the standalone exporter with the correct parameters.

## Requirements

- Python 3.9+
- Pillow (automatically installed with `pip install -r requirements.txt`)
- No display required for standalone exporter!

## Troubleshooting

- **"No display" error**: Use the standalone `export_gif.py` script instead
- **Large file sizes**: Use `--mode simple` for 90% smaller files
- **Slow generation**: Reduce `--max-steps` or grid size (`--H`, `--W`)
- **Memory issues**: Close other applications, use smaller episodes

---

Happy GIF making! Share your agent's adventures! 🤖✨