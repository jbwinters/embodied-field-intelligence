# GitHub Pages Research Site

A complete research website for the Embodied Field Intelligence project has been created in the `docs/` directory.

## 📊 What's Included

### 1. **Complete Research Website** (`docs/index.html`)
- Professional landing page with hero section and demo GIF
- Technical approach explanation with field dynamics
- Interactive charts showing experimental results
- Code examples and future directions

### 2. **Data Analysis**
- **Baseline Performance**: Multi-seed evaluation showing agent consistency
- **Ablation Study**: Component-wise analysis showing each field's contribution
- **Scaling Analysis**: Performance across different grid sizes (10×10 to 30×30)
- **Parameter Sensitivity**: Optimal diffusion rate identification

### 3. **Visualizations**
- Animated GIF demos of agent navigation
- Interactive charts using Chart.js
- Performance comparison graphs
- Statistical analysis plots

## 📈 Key Findings from Analysis

1. **Field Contributions** (Ablation Study):
   - Full model: -0.92 ± 0.95 return
   - Without trail (worst): -2.00 ± 0.00 return
   - Without novelty: -0.88 ± 0.95 return
   - Trail field is critical for exploration

2. **Scaling Performance**:
   - Performance improves with larger grids
   - 30×30 grids achieve positive returns (+0.8)
   - Larger environments provide more navigation opportunities

3. **Parameter Sensitivity**:
   - Optimal diffusion rate: 0.25
   - Too low (<0.1): Limited field spread
   - Too high (>0.3): Loss of gradient precision

## 🚀 Deployment Instructions

### Step 1: Push to GitHub
```bash
git add docs/
git commit -m "Add GitHub Pages research site"
git push origin main
```

### Step 2: Enable GitHub Pages
1. Go to your repository Settings
2. Navigate to Pages section
3. Source: Deploy from a branch
4. Branch: `main` → folder: `/docs`
5. Save

### Step 3: Access Your Site
After a few minutes, your site will be live at:
```
https://[your-username].github.io/embodied_field_intelligence/
```

## 🛠️ Local Testing

```bash
# Navigate to docs folder
cd docs/

# Start local server
python -m http.server 8080

# Open in browser
# http://localhost:8080
```

## 📁 File Structure

```
docs/
├── index.html           # Main research page
├── _config.yml         # Jekyll configuration
├── README.md           # Documentation
└── assets/
    ├── css/
    │   └── style.css   # Professional styling
    ├── js/
    │   └── main.js     # Interactive charts
    ├── data/
    │   └── analysis_results.json  # Experimental data
    └── images/
        ├── efi_simple_*.gif       # Demo animations
        ├── analysis_charts.png    # Ablation/scaling plots
        └── sensitivity_plot.png   # Parameter analysis
```

## 🎨 Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Scrolling**: Navigation with anchor links
- **Interactive Charts**: Real-time data visualization
- **Fade-in Animations**: Elements appear as you scroll
- **Professional Styling**: Modern gradient hero, card layouts
- **Code Examples**: Syntax-highlighted command snippets

## 📝 Customization

To update the site with new data:

1. **Run new experiments**:
```bash
python run_analysis.py
```

2. **Generate new demo GIFs**:
```bash
python export_gif.py --seed 42 --H 25 --W 25 --mode simple --output-dir docs/assets/images
```

3. **Update content**: Edit `docs/index.html` directly

4. **Modify styling**: Edit `docs/assets/css/style.css`

5. **Change charts**: Edit `docs/assets/js/main.js`

## 🔬 Research Highlights

The site presents EFI as a novel approach to embodied AI that:
- Uses cellular automata instead of neural networks
- Implements intelligence through field dynamics
- Provides interpretable decision-making
- Achieves continuous adaptation without training phases

Perfect for sharing your research with the community!