"""Interactive viewer with play/pause/step controls."""

from typing import List, Optional, Callable
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation


class InteractiveViewer:
    """
    Interactive viewer for episode playback with controls.
    
    Features:
    - Play/Pause
    - Step forward/backward
    - Speed control
    - Frame slider
    - Multi-panel visualization
    """
    
    def __init__(self, 
                 frames: List[dict],
                 world_frames: List[np.ndarray],
                 fps: int = 8,
                 title: str = "EFI Interactive Viewer"):
        """
        Initialize interactive viewer.
        
        Args:
            frames: List of field dictionaries per timestep
            world_frames: List of RGB world images
            fps: Target frames per second
            title: Window title
        """
        self.frames = frames
        self.world_frames = world_frames
        self.n_frames = len(frames)
        self.fps = fps
        self.title = title
        
        # State
        self.current_frame = 0
        self.playing = False
        self.speed_multiplier = 1.0
        
        # Create figure and axes
        self._setup_figure()
        
        # Animation (initially paused)
        self.animation = None
        
    def _setup_figure(self):
        """Setup the figure with panels and controls."""
        # Create figure with GridSpec for better layout control
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(self.title, fontsize=14)
        
        # Create grid: 3 rows, 4 columns for fields + 1 row for controls
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(4, 4, figure=self.fig, height_ratios=[1, 1, 1, 0.15])
        
        # Field displays (3x4 grid)
        self.axes = {}
        self.ims = {}
        
        # Row 1
        self.axes['world'] = self.fig.add_subplot(gs[0, 0])
        self.axes['GA'] = self.fig.add_subplot(gs[0, 1])
        self.axes['GB'] = self.fig.add_subplot(gs[0, 2])
        self.axes['P_eff'] = self.fig.add_subplot(gs[0, 3])
        
        # Row 2
        self.axes['Vtrail'] = self.fig.add_subplot(gs[1, 0])
        self.axes['Novel'] = self.fig.add_subplot(gs[1, 1])
        self.axes['Ssum'] = self.fig.add_subplot(gs[1, 2])
        self.axes['info'] = self.fig.add_subplot(gs[1, 3])
        
        # Initialize images
        self._init_images()
        
        # Control panel (bottom row, spanning all columns)
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.axis('off')
        
        # Add controls
        self._add_controls()
        
        # Frame slider (row 2, spanning all columns)
        self.ax_slider = self.fig.add_subplot(gs[2, :])
        self.ax_slider.axis('off')
        self._add_slider()
        
        plt.tight_layout()
        
    def _init_images(self):
        """Initialize image displays."""
        # Get first frame to setup
        frame = self.frames[0] if self.frames else None
        world = self.world_frames[0] if self.world_frames else np.zeros((10,10,3), dtype=np.uint8)
        
        # World view
        ax = self.axes['world']
        self.ims['world'] = ax.imshow(world)
        ax.set_title("World")
        ax.axis('off')
        
        if frame:
            # GA field
            ax = self.axes['GA']
            self.ims['GA'] = ax.imshow(frame.get('GA', np.zeros((10,10))), cmap='Greens', vmin=0, vmax=1)
            ax.set_title("GA (A scent)")
            ax.axis('off')
            
            # GB field
            ax = self.axes['GB']
            self.ims['GB'] = ax.imshow(frame.get('GB', np.zeros((10,10))), cmap='Purples', vmin=0, vmax=1)
            ax.set_title("GB (B scent)")
            ax.axis('off')
            
            # P_eff field
            ax = self.axes['P_eff']
            self.ims['P_eff'] = ax.imshow(frame.get('P_eff', np.zeros((10,10))), cmap='plasma')
            ax.set_title("P_eff (potential)")
            ax.axis('off')
            
            # Visit trail
            ax = self.axes['Vtrail']
            self.ims['Vtrail'] = ax.imshow(frame.get('Vtrail', np.zeros((10,10))), cmap='Oranges', vmin=0, vmax=1)
            ax.set_title("Visit trail")
            ax.axis('off')
            
            # Novelty
            ax = self.axes['Novel']
            self.ims['Novel'] = ax.imshow(frame.get('Novel', np.zeros((10,10))), cmap='Blues', vmin=0, vmax=1)
            ax.set_title("Novelty")
            ax.axis('off')
            
            # Schema sum
            ax = self.axes['Ssum']
            self.ims['Ssum'] = ax.imshow(frame.get('Ssum', np.zeros((10,10))), cmap='cividis', vmin=0)
            ax.set_title("Schema sum")
            ax.axis('off')
        
        # Info panel
        ax = self.axes['info']
        ax.axis('off')
        self.info_text = ax.text(0.05, 0.5, "", fontsize=10, transform=ax.transAxes, 
                                 verticalalignment='center', family='monospace')
        
    def _add_controls(self):
        """Add control buttons."""
        # Button dimensions
        btn_width = 0.08
        btn_height = 0.04
        y_pos = 0.05
        
        # Play/Pause button
        self.btn_play = Button(
            plt.axes([0.1, y_pos, btn_width, btn_height]),
            'Play'
        )
        self.btn_play.on_clicked(self._on_play_pause)
        
        # Step backward
        self.btn_prev = Button(
            plt.axes([0.2, y_pos, btn_width, btn_height]),
            '← Prev'
        )
        self.btn_prev.on_clicked(self._on_prev)
        
        # Step forward
        self.btn_next = Button(
            plt.axes([0.3, y_pos, btn_width, btn_height]),
            'Next →'
        )
        self.btn_next.on_clicked(self._on_next)
        
        # Reset button
        self.btn_reset = Button(
            plt.axes([0.4, y_pos, btn_width, btn_height]),
            'Reset'
        )
        self.btn_reset.on_clicked(self._on_reset)
        
        # Speed control
        self.slider_speed = Slider(
            plt.axes([0.55, y_pos, 0.15, btn_height]),
            'Speed', 0.25, 4.0, valinit=1.0, valstep=0.25
        )
        self.slider_speed.on_changed(self._on_speed_change)
        
    def _add_slider(self):
        """Add frame slider."""
        self.slider_frame = Slider(
            plt.axes([0.15, 0.02, 0.7, 0.03]),
            'Frame', 0, max(0, self.n_frames - 1), 
            valinit=0, valstep=1, valfmt='%d'
        )
        self.slider_frame.on_changed(self._on_frame_change)
        
    def _update_frame(self, frame_idx: int):
        """Update display to show specific frame."""
        if not 0 <= frame_idx < self.n_frames:
            return
            
        self.current_frame = frame_idx
        
        # Update slider position (without triggering callback)
        self.slider_frame.set_val(frame_idx)
        
        # Update world image
        if frame_idx < len(self.world_frames):
            self.ims['world'].set_data(self.world_frames[frame_idx])
        
        # Update field images
        if frame_idx < len(self.frames):
            frame = self.frames[frame_idx]
            
            for field_name in ['GA', 'GB', 'P_eff', 'Vtrail', 'Novel', 'Ssum']:
                if field_name in frame and field_name in self.ims:
                    self.ims[field_name].set_data(frame[field_name])
        
        # Update info text
        info = frame.get('info', {}) if frame_idx < len(self.frames) else {}
        info_lines = [
            f"Frame: {frame_idx + 1}/{self.n_frames}",
            f"Step: {info.get('step', frame_idx)}",
            f"Return: {info.get('return', 0.0):+.3f}",
            f"Action: {info.get('action', 'N/A')}",
            f"Stuck: {info.get('stuck_count', 0)}",
        ]
        self.info_text.set_text('\n'.join(info_lines))
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def _on_play_pause(self, event):
        """Handle play/pause button."""
        self.playing = not self.playing
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')
        
        if self.playing:
            self._start_animation()
        else:
            self._stop_animation()
            
    def _on_prev(self, event):
        """Handle previous frame button."""
        self.playing = False
        self.btn_play.label.set_text('Play')
        self._stop_animation()
        
        if self.current_frame > 0:
            self._update_frame(self.current_frame - 1)
            
    def _on_next(self, event):
        """Handle next frame button."""
        self.playing = False
        self.btn_play.label.set_text('Play')
        self._stop_animation()
        
        if self.current_frame < self.n_frames - 1:
            self._update_frame(self.current_frame + 1)
            
    def _on_reset(self, event):
        """Handle reset button."""
        self.playing = False
        self.btn_play.label.set_text('Play')
        self._stop_animation()
        self._update_frame(0)
        
    def _on_speed_change(self, val):
        """Handle speed slider change."""
        self.speed_multiplier = val
        if self.playing and self.animation:
            # Restart animation with new speed
            self._stop_animation()
            self._start_animation()
            
    def _on_frame_change(self, val):
        """Handle frame slider change."""
        frame_idx = int(val)
        if frame_idx != self.current_frame:
            self._update_frame(frame_idx)
            
    def _start_animation(self):
        """Start animation playback."""
        if self.animation:
            self._stop_animation()
            
        interval = int(1000.0 / (self.fps * self.speed_multiplier))
        
        def animate(i):
            if self.playing:
                next_frame = (self.current_frame + 1) % self.n_frames
                self._update_frame(next_frame)
                if next_frame == 0:  # Loop completed
                    self.playing = False
                    self.btn_play.label.set_text('Play')
            return []
        
        self.animation = FuncAnimation(
            self.fig, animate, 
            interval=interval, 
            blit=False, 
            repeat=True
        )
        
    def _stop_animation(self):
        """Stop animation playback."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            
    def show(self):
        """Display the interactive viewer."""
        # Update to first frame
        self._update_frame(0)
        
        # Show window
        plt.show()
        
        
def create_interactive_viewer(episode_data: dict) -> InteractiveViewer:
    """
    Create interactive viewer from episode data.
    
    Args:
        episode_data: Dictionary with 'frames' and 'world_frames' lists
        
    Returns:
        InteractiveViewer instance
    """
    frames = episode_data.get('frames', [])
    world_frames = episode_data.get('world_frames', [])
    
    viewer = InteractiveViewer(frames, world_frames)
    return viewer