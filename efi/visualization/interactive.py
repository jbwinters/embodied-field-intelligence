"""Interactive viewer with play/pause/step controls."""

from typing import List, Optional, Callable
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation, PillowWriter


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
                 title: str = "EFI Interactive Viewer",
                 final_metrics: Optional[dict] = None):
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
        self.final_metrics = final_metrics or {}
        
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
        
        # Row 3 - Affect fields (if present)
        self.axes['Pain'] = self.fig.add_subplot(gs[2, 0])
        self.axes['Membrane'] = self.fig.add_subplot(gs[2, 1])
        
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
            
            # Pain field (if present)
            ax = self.axes['Pain']
            if 'Pain' in frame:
                self.ims['Pain'] = ax.imshow(frame.get('Pain', np.zeros((10,10))), cmap='Reds', vmin=0, vmax=1)
                ax.set_title("Pain field")
            else:
                ax.text(0.5, 0.5, "Pain\n(not enabled)", ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Membrane field (if present)
            ax = self.axes['Membrane']
            if 'Membrane' in frame:
                self.ims['Membrane'] = ax.imshow(frame.get('Membrane', np.zeros((10,10))), cmap='YlOrBr', vmin=0, vmax=1)
                ax.set_title("Membrane field")
            else:
                ax.text(0.5, 0.5, "Membrane\n(not enabled)", ha='center', va='center', transform=ax.transAxes)
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
        
        # Export GIF button
        self.btn_export = Button(
            plt.axes([0.75, y_pos, btn_width, btn_height]),
            'Export GIF'
        )
        self.btn_export.on_clicked(self._on_export_gif)
        
        # Export Simple GIF button  
        self.btn_export_simple = Button(
            plt.axes([0.85, y_pos, btn_width, btn_height]),
            'Simple GIF'
        )
        self.btn_export_simple.on_clicked(self._on_export_simple_gif)
        
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
            
            for field_name in ['GA', 'GB', 'P_eff', 'Vtrail', 'Novel', 'Ssum', 'Pain', 'Membrane']:
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
        
        # Add affect info if present
        if 'pain' in info:
            info_lines.append(f"Pain: {info.get('pain', 0.0):.3f}")
        if 'arousal' in info:
            info_lines.append(f"Arousal: {info.get('arousal', 0.0):.3f}")
        if 'learning_gate' in info:
            info_lines.append(f"Learn Gate: {info.get('learning_gate', 1.0):.3f}")
        
        # Add final metrics on last frame
        if frame_idx == self.n_frames - 1 and self.final_metrics:
            info_lines.append("\n=== Final Metrics ===")
            if 'coverage' in self.final_metrics:
                info_lines.append(f"Coverage: {self.final_metrics['coverage']:.1%}")
            if 'frontier_efficiency' in self.final_metrics:
                info_lines.append(f"Frontier Eff: {self.final_metrics['frontier_efficiency']:.3f}")
            if 'path_optimality' in self.final_metrics and self.final_metrics['path_optimality']:
                info_lines.append(f"Path Opt: {self.final_metrics['path_optimality']:.1f}x")
            if 'backtrack_rate' in self.final_metrics:
                info_lines.append(f"Backtrack: {self.final_metrics['backtrack_rate']:.1%}")
        
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
    
    def _on_export_gif(self, event):
        """Handle export GIF button."""
        # Create output directory if it doesn't exist
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = output_dir / f"efi_episode_{timestamp}.gif"
        
        # Temporarily disable interactive elements
        self.btn_export.label.set_text('Exporting...')
        self.btn_export.ax.set_facecolor('#ffcccc')
        self.fig.canvas.draw_idle()
        
        try:
            # Create a new figure for the GIF with composite layout
            self._export_gif_to_file(gif_path)
            
            # Show success message
            self.btn_export.label.set_text('Export GIF')
            self.btn_export.ax.set_facecolor('#ccffcc')
            print(f"GIF exported successfully to: {gif_path}")
            
            # Reset button color after a moment
            self.fig.canvas.draw_idle()
            plt.pause(1.0)
            self.btn_export.ax.set_facecolor('#f0f0f0')
            
        except Exception as e:
            print(f"Error exporting GIF: {e}")
            self.btn_export.label.set_text('Export Failed')
            self.btn_export.ax.set_facecolor('#ffcccc')
        
        self.fig.canvas.draw_idle()
    
    def _export_gif_to_file(self, filepath: Path):
        """Export the episode as an animated GIF."""
        # Create a new figure for the export
        export_fig = plt.figure(figsize=(12, 8))
        export_fig.suptitle(f"EFI Episode - {len(self.frames)} frames", fontsize=12)
        
        # Create grid layout
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 3, figure=export_fig, height_ratios=[1, 1])
        
        # Create axes for export
        axes = {}
        axes['world'] = export_fig.add_subplot(gs[0, 0])
        axes['GA'] = export_fig.add_subplot(gs[0, 1])
        axes['GB'] = export_fig.add_subplot(gs[0, 2])
        axes['P_eff'] = export_fig.add_subplot(gs[1, 0])
        axes['Vtrail'] = export_fig.add_subplot(gs[1, 1])
        axes['info'] = export_fig.add_subplot(gs[1, 2])
        
        # Initialize images for export
        ims = {}
        
        # Setup world view
        ax = axes['world']
        world = self.world_frames[0] if self.world_frames else np.zeros((10,10,3), dtype=np.uint8)
        ims['world'] = ax.imshow(world)
        ax.set_title("World", fontsize=10)
        ax.axis('off')
        
        if self.frames:
            frame = self.frames[0]
            
            # GA field
            ax = axes['GA']
            ims['GA'] = ax.imshow(frame.get('GA', np.zeros((10,10))), cmap='Greens', vmin=0, vmax=1)
            ax.set_title("GA (A scent)", fontsize=10)
            ax.axis('off')
            
            # GB field  
            ax = axes['GB']
            ims['GB'] = ax.imshow(frame.get('GB', np.zeros((10,10))), cmap='Purples', vmin=0, vmax=1)
            ax.set_title("GB (B scent)", fontsize=10)
            ax.axis('off')
            
            # P_eff field
            ax = axes['P_eff']
            ims['P_eff'] = ax.imshow(frame.get('P_eff', np.zeros((10,10))), cmap='plasma')
            ax.set_title("Effective Potential", fontsize=10)
            ax.axis('off')
            
            # Visit trail
            ax = axes['Vtrail']
            ims['Vtrail'] = ax.imshow(frame.get('Vtrail', np.zeros((10,10))), cmap='Oranges', vmin=0, vmax=1)
            ax.set_title("Visit Trail", fontsize=10)
            ax.axis('off')
        
        # Info panel
        ax = axes['info']
        ax.axis('off')
        info_text = ax.text(0.05, 0.5, "", fontsize=9, transform=ax.transAxes,
                           verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Animation update function
        def update(frame_idx):
            # Update world
            if frame_idx < len(self.world_frames):
                ims['world'].set_data(self.world_frames[frame_idx])
            
            # Update fields
            if frame_idx < len(self.frames):
                frame = self.frames[frame_idx]
                
                for field_name in ['GA', 'GB', 'P_eff', 'Vtrail']:
                    if field_name in frame and field_name in ims:
                        ims[field_name].set_data(frame[field_name])
                
                # Update info text
                info = frame.get('info', {})
                info_lines = [
                    f"Frame: {frame_idx + 1}/{self.n_frames}",
                    f"Step: {info.get('step', frame_idx)}",
                    f"Return: {info.get('return', 0.0):+.3f}",
                    f"Action: {info.get('action', 'N/A')}",
                ]
                info_text.set_text('\n'.join(info_lines))
            
            return list(ims.values()) + [info_text]
        
        # Create animation
        anim = FuncAnimation(
            export_fig, update,
            frames=self.n_frames,
            interval=int(1000.0 / self.fps),
            blit=True
        )
        
        # Save as GIF using Pillow
        writer = PillowWriter(fps=self.fps)
        anim.save(str(filepath), writer=writer, dpi=80)
        
        # Clean up
        plt.close(export_fig)
    
    def _on_export_simple_gif(self, event):
        """Handle export simple GIF button (world view only)."""
        # Create output directory if it doesn't exist
        output_dir = Path("exports")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = output_dir / f"efi_world_{timestamp}.gif"
        
        # Temporarily disable interactive elements
        self.btn_export_simple.label.set_text('Exporting...')
        self.btn_export_simple.ax.set_facecolor('#ffcccc')
        self.fig.canvas.draw_idle()
        
        try:
            # Create a simple world-only GIF
            self._export_simple_gif_to_file(gif_path)
            
            # Show success message
            self.btn_export_simple.label.set_text('Simple GIF')
            self.btn_export_simple.ax.set_facecolor('#ccffcc')
            print(f"Simple GIF exported successfully to: {gif_path}")
            
            # Reset button color after a moment
            self.fig.canvas.draw_idle()
            plt.pause(1.0)
            self.btn_export_simple.ax.set_facecolor('#f0f0f0')
            
        except Exception as e:
            print(f"Error exporting simple GIF: {e}")
            self.btn_export_simple.label.set_text('Export Failed')
            self.btn_export_simple.ax.set_facecolor('#ffcccc')
        
        self.fig.canvas.draw_idle()
    
    def _export_simple_gif_to_file(self, filepath: Path):
        """Export a simple world-only GIF for sharing."""
        # Create a new figure for the export
        export_fig = plt.figure(figsize=(6, 6))
        
        # Single axis for world view
        ax = export_fig.add_subplot(111)
        world = self.world_frames[0] if self.world_frames else np.zeros((10,10,3), dtype=np.uint8)
        im = ax.imshow(world)
        ax.set_title("EFI Agent Navigation", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add frame counter
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Animation update function
        def update(frame_idx):
            # Update world
            if frame_idx < len(self.world_frames):
                im.set_data(self.world_frames[frame_idx])
            
            # Update frame counter
            if frame_idx < len(self.frames):
                info = self.frames[frame_idx].get('info', {})
                frame_text.set_text(f"Step {frame_idx+1}/{self.n_frames} | Score: {info.get('return', 0.0):+.2f}")
            
            return [im, frame_text]
        
        # Create animation
        anim = FuncAnimation(
            export_fig, update,
            frames=self.n_frames,
            interval=int(1000.0 / self.fps),
            blit=True
        )
        
        # Save as GIF using Pillow with optimized settings for smaller file size
        writer = PillowWriter(fps=self.fps)
        anim.save(str(filepath), writer=writer, dpi=60)
        
        # Clean up
        plt.close(export_fig)
            
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
    final_metrics = episode_data.get('final_metrics', {})
    
    viewer = InteractiveViewer(frames, world_frames, final_metrics=final_metrics)
    return viewer