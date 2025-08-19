"""HTML-based interactive viewer for headless environments."""

import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def array_to_base64_png(array: np.ndarray, cmap: str = None, vmin: float = None, vmax: float = None) -> str:
    """Convert numpy array to base64-encoded PNG."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis('off')
    
    if array.ndim == 3:  # RGB image
        ax.imshow(array)
    else:  # 2D field
        ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.tight_layout(pad=0)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close(fig)
    
    # Encode to base64
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"


def create_html_viewer(episode_data: dict, output_path: str = "interactive_viewer.html") -> str:
    """
    Create an HTML interactive viewer from episode data.
    
    Args:
        episode_data: Dictionary with 'frames' and 'world_frames'
        output_path: Path to save HTML file
        
    Returns:
        Path to created HTML file
    """
    frames = episode_data.get('frames', [])
    world_frames = episode_data.get('world_frames', [])
    metrics = episode_data.get('metrics', {})
    
    # Convert metrics object to dict if needed
    if hasattr(metrics, '__dict__'):
        metrics = vars(metrics)
    elif not isinstance(metrics, dict):
        metrics = {}
    
    if not frames:
        print("No frames to display")
        return None
    
    # Convert frames to base64 images
    frame_data = []
    for i, (frame, world) in enumerate(zip(frames, world_frames)):
        frame_images = {
            'world': array_to_base64_png(world),
            'GA': array_to_base64_png(frame.get('GA', np.zeros((10,10))), cmap='Greens', vmin=0, vmax=1),
            'GB': array_to_base64_png(frame.get('GB', np.zeros((10,10))), cmap='Purples', vmin=0, vmax=1),
            'P_eff': array_to_base64_png(frame.get('P_eff', np.zeros((10,10))), cmap='plasma'),
            'Vtrail': array_to_base64_png(frame.get('Vtrail', np.zeros((10,10))), cmap='Oranges', vmin=0, vmax=1),
            'Novel': array_to_base64_png(frame.get('Novel', np.zeros((10,10))), cmap='Blues', vmin=0, vmax=1),
            'Ssum': array_to_base64_png(frame.get('Ssum', np.zeros((10,10))), cmap='cividis'),
        }
        
        # Add affect fields if present
        if 'Pain' in frame:
            frame_images['Pain'] = array_to_base64_png(frame['Pain'], cmap='Reds', vmin=0, vmax=1)
        if 'Membrane' in frame:
            frame_images['Membrane'] = array_to_base64_png(frame['Membrane'], cmap='YlOrBr', vmin=0, vmax=1)
        
        info = frame.get('info', {})
        frame_data.append({
            'images': frame_images,
            'info': info
        })
    
    # Create HTML
    html_template = '''<!DOCTYPE html>
<html>
<head>
    <title>EFI Interactive Viewer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1e1e1e;
            color: #e0e0e0;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .controls {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        button:hover {
            background: #45a049;
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        input[type="range"] {
            width: 200px;
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .panel {
            background: #2d2d2d;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        .panel h3 {
            margin: 0 0 10px 0;
            color: #4CAF50;
            font-size: 14px;
        }
        .panel img {
            width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .info {
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .frame-slider {
            width: 100%;
            margin: 20px 0;
        }
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .label {
            color: #999;
            font-size: 12px;
        }
        .value {
            color: #4CAF50;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>EFI Interactive Episode Viewer</h1>
        
        <div class="controls">
            <button id="playBtn" onclick="togglePlay()">▶ Play</button>
            <button onclick="prevFrame()">⏮ Prev</button>
            <button onclick="nextFrame()">⏭ Next</button>
            <button onclick="reset()">⏹ Reset</button>
            <button onclick="exportGIF()">📥 Export GIF</button>
            <button onclick="exportSimpleGIF()">📷 Simple GIF</button>
            
            <div class="slider-container">
                <span class="label">Speed:</span>
                <input type="range" id="speedSlider" min="0.25" max="4" step="0.25" value="1">
                <span id="speedLabel" class="value">1x</span>
            </div>
            
            <div class="status">
                <div><span class="label">Frame:</span> <span id="frameNum" class="value">1</span>/<span id="totalFrames">0</span></div>
                <div><span class="label">Step:</span> <span id="stepNum" class="value">0</span></div>
                <div><span class="label">Return:</span> <span id="returnVal" class="value">0.000</span></div>
            </div>
        </div>
        
        <input type="range" id="frameSlider" class="frame-slider" min="0" max="0" value="0">
        
        <div class="grid">
            <div class="panel">
                <h3>World</h3>
                <img id="img_world" src="">
            </div>
            <div class="panel">
                <h3>GA (A scent)</h3>
                <img id="img_GA" src="">
            </div>
            <div class="panel">
                <h3>GB (B scent)</h3>
                <img id="img_GB" src="">
            </div>
            <div class="panel">
                <h3>P_eff (potential)</h3>
                <img id="img_P_eff" src="">
            </div>
            <div class="panel">
                <h3>Visit Trail</h3>
                <img id="img_Vtrail" src="">
            </div>
            <div class="panel">
                <h3>Novelty</h3>
                <img id="img_Novel" src="">
            </div>
            <div class="panel">
                <h3>Schema Sum</h3>
                <img id="img_Ssum" src="">
            </div>
            <div class="panel">
                <h3>Info</h3>
                <div class="info" id="infoText">
                    Loading...
                </div>
            </div>
            <div class="panel" id="painPanel" style="display:none">
                <h3>Pain Field</h3>
                <img id="img_Pain" src="">
            </div>
            <div class="panel" id="membranePanel" style="display:none">
                <h3>Membrane Field</h3>
                <img id="img_Membrane" src="">
            </div>
        </div>
    </div>
    
    <script>
        // Frame data embedded
        const frameData = ''' + json.dumps(frame_data) + ''';
        
        let currentFrame = 0;
        let playing = false;
        let playInterval = null;
        let speed = 1.0;
        
        // Initialize
        document.getElementById('totalFrames').textContent = frameData.length;
        document.getElementById('frameSlider').max = frameData.length - 1;
        
        // Speed slider
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            speed = parseFloat(e.target.value);
            document.getElementById('speedLabel').textContent = speed + 'x';
            if (playing) {
                stopPlay();
                startPlay();
            }
        });
        
        // Frame slider
        document.getElementById('frameSlider').addEventListener('input', (e) => {
            currentFrame = parseInt(e.target.value);
            updateDisplay();
        });
        
        function updateDisplay() {
            if (currentFrame < 0 || currentFrame >= frameData.length) return;
            
            const frame = frameData[currentFrame];
            
            // Update images
            for (const [key, src] of Object.entries(frame.images)) {
                const img = document.getElementById('img_' + key);
                if (img) img.src = src;
            }
            
            // Show/hide affect panels
            const painPanel = document.getElementById('painPanel');
            const membranePanel = document.getElementById('membranePanel');
            if (frame.images.Pain) {
                painPanel.style.display = 'block';
            }
            if (frame.images.Membrane) {
                membranePanel.style.display = 'block';
            }
            
            // Update info
            const info = frame.info || {};
            document.getElementById('frameNum').textContent = currentFrame + 1;
            document.getElementById('stepNum').textContent = info.step || 0;
            document.getElementById('returnVal').textContent = (info.return || 0).toFixed(3);
            
            // Update info text
            const infoLines = [
                `Action: ${info.action !== undefined ? info.action : 'N/A'}`,
                `Reward: ${info.reward !== undefined ? info.reward.toFixed(3) : 'N/A'}`,
                `Stuck: ${info.stuck_count || 0}`
            ];
            
            // Add affect info if present
            if (info.pain !== undefined) {
                infoLines.push(`Pain: ${info.pain.toFixed(3)}`);
            }
            if (info.arousal !== undefined) {
                infoLines.push(`Arousal: ${info.arousal.toFixed(3)}`);
            }
            if (info.learning_gate !== undefined) {
                infoLines.push(`Learn Gate: ${info.learning_gate.toFixed(3)}`);
            }
            
            document.getElementById('infoText').textContent = infoLines.join('\\n');
            
            // Update slider
            document.getElementById('frameSlider').value = currentFrame;
        }
        
        function togglePlay() {
            if (playing) {
                stopPlay();
            } else {
                startPlay();
            }
        }
        
        function startPlay() {
            playing = true;
            document.getElementById('playBtn').textContent = '⏸ Pause';
            playInterval = setInterval(() => {
                if (currentFrame >= frameData.length - 1) {
                    currentFrame = 0;
                } else {
                    currentFrame++;
                }
                updateDisplay();
            }, 1000 / (8 * speed));
        }
        
        function stopPlay() {
            playing = false;
            document.getElementById('playBtn').textContent = '▶ Play';
            if (playInterval) {
                clearInterval(playInterval);
                playInterval = null;
            }
        }
        
        function nextFrame() {
            stopPlay();
            if (currentFrame < frameData.length - 1) {
                currentFrame++;
                updateDisplay();
            }
        }
        
        function prevFrame() {
            stopPlay();
            if (currentFrame > 0) {
                currentFrame--;
                updateDisplay();
            }
        }
        
        function reset() {
            stopPlay();
            currentFrame = 0;
            updateDisplay();
        }
        
        function exportGIF() {
            alert('GIF Export Instructions:\\n\\n' +
                  'Use the standalone GIF exporter:\\n\\n' +
                  'python export_gif.py --mode full\\n\\n' +
                  'Add parameters to match your episode:\\n' +
                  '--seed [seed] --H [height] --W [width] --max-steps [steps]\\n' +
                  '--nA [A_targets] --nB [B_targets]\\n\\n' +
                  'This will re-run the episode and export a full multi-panel GIF.\\n' +
                  'The GIF will be saved in the exports/ directory.');
        }
        
        function exportSimpleGIF() {
            alert('Simple GIF Export Instructions:\\n\\n' +
                  'Use the standalone GIF exporter:\\n\\n' +
                  'python export_gif.py --mode simple\\n\\n' +
                  'Add parameters to match your episode:\\n' +
                  '--seed [seed] --H [height] --W [width] --max-steps [steps]\\n' +
                  '--nA [A_targets] --nB [B_targets]\\n\\n' +
                  'This will re-run the episode and export a simple world-only GIF.\\n' +
                  'The GIF will be saved in the exports/ directory.\\n\\n' +
                  'Note: Simple GIFs are ~10x smaller and better for sharing!');
        }
        
        // Initial display
        updateDisplay();
    </script>
</body>
</html>'''
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.write_text(html_template)
    
    return str(output_path.absolute())


def save_episode_as_html(episode_data: dict, output_dir: str = "runs") -> str:
    """
    Save episode data as an HTML viewer file.
    
    Args:
        episode_data: Episode data with frames
        output_dir: Directory to save HTML file
        
    Returns:
        Path to created HTML file
    """
    from ..core import ensure_dir, ts
    
    output_dir = ensure_dir(output_dir)
    filename = f"interactive_episode_{ts()}.html"
    output_path = output_dir / filename
    
    return create_html_viewer(episode_data, str(output_path))