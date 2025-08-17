"""Video generation utilities."""

from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ..core import ensure_dir


def save_video_mp4(frames: List[np.ndarray], path: Path, fps: int = 8):
    """
    Save frames as MP4 video.
    
    Args:
        frames: List of RGB frames
        path: Output video path
        fps: Frames per second
    """
    if len(frames) == 0:
        return
        
    try:
        from matplotlib.animation import FFMpegWriter
        
        ensure_dir(path.parent)
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        ax.axis('off')
        im = ax.imshow(frames[0])
        
        writer = FFMpegWriter(fps=fps)
        with writer.saving(fig, str(path), dpi=150):
            for fr in frames:
                im.set_data(fr)
                writer.grab_frame()
                
        plt.close(fig)
        print(f"[video] saved to {path}")
        
    except Exception as e:
        print(f"[video] ffmpeg not available, falling back to frame sequence")
        save_frame_sequence(frames, path.with_suffix(""))


def save_frame_sequence(frames: List[np.ndarray], output_dir: Path):
    """
    Save frames as PNG sequence.
    
    Args:
        frames: List of RGB frames
        output_dir: Directory for frame sequence
    """
    if len(frames) == 0:
        return
        
    seq_dir = ensure_dir(output_dir.as_posix() + "_frames")
    
    for i, fr in enumerate(frames):
        plt.imsave(seq_dir / f"frame_{i:05d}.png", fr)
        
    print(f"[video] saved PNG sequence to {seq_dir}")