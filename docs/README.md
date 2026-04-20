# EFI Research Website

This is the GitHub Pages site for the Embodied Field Intelligence research project.

## Viewing the Site

### Online
Once pushed to GitHub and GitHub Pages is enabled, the site will be available at:
```
https://jbwinters.github.io/embodied-field-intelligence/
```

### Local Development
To view the site locally:

```bash
# Simple Python server
cd docs/
python -m http.server 8000

# Then open http://localhost:8000 in your browser
```

## Structure

- `index.html` - Main research overview page
- `assets/css/` - Styling
- `assets/js/` - Interactive charts and animations
- `assets/data/` - Analysis results in JSON format
- `assets/images/` - GIFs and visualizations

## Data Updates

To update the site artifacts:

1. Regenerate analysis data from the project root:
```bash
python scripts/run_analysis.py
```

2. Refresh the charts from the generated data:
```bash
python scripts/analyze_results.py
```

3. Generate new demo GIFs:
```bash
python scripts/export_gif.py --seed 42 --mode simple --output-dir docs/assets/images
```

4. Commit and push changes

## Enabling GitHub Pages

1. Push this repository to GitHub
2. Go to Settings → Pages
3. Set Source to "Deploy from a branch"
4. Select `main` branch and `/docs` folder
5. Save and wait a few minutes for deployment

## Customization

- Edit `index.html` to update content
- Modify `assets/css/style.css` for styling changes
- Update `assets/js/main.js` to change chart configurations
- Replace GIFs in `assets/images/` with your own demos

## Analysis Results

The page displays:
- Ablation study showing component contributions
- Scaling analysis across different grid sizes
- Parameter sensitivity for diffusion rates
- Performance statistics and key findings

Results are loaded from `assets/data/experiment_summary.json` with fallback data if the file is unavailable.