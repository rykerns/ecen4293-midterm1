# Wraps the PDE solver and handles visualization + animation

import solver as sv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import config as con
import numpy as np

def run_and_animate(cfg):
    """
    Run the solver with given config and show an interactive animation.
    This function is called from config.py or main entry point.

    Controls:
        Space:      Play / Pause
        Up:         +10 frames forward
        Down:       -10 frames backward
        A:          -100 frames backward
        D:          +100 frames forward
    """
    # ─── Run simulation ──────────────────────────────────────────────────────
    result = sv.run(cfg)
    snapshots = result.snapshots
    times = result.times

    n_frames = len(snapshots)

    # Calculate real time step between stored frames for accurate controls text
    if n_frames > 1:
        dt_frame = times[1] - times[0]
        time_10  = 10 * dt_frame
        time_100 = 100 * dt_frame
    else:
        time_10 = time_100 = 0.0

    print(f"Simulation complete. Frames: {n_frames}, Time range: {times[0]:.3f} → {times[-1]:.3f}")
    print(f"Initial min/max: {snapshots[0].min():.2f} / {snapshots[0].max():.2f}")
    print(f"Final min/max:   {snapshots[-1].min():.2f} / {snapshots[-1].max():.2f}")

    Lx = result.metadata["Lx"]
    Ly = result.metadata["Ly"]

    # ─── Create figure and main axes ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))

    img = ax.imshow(
        snapshots[0],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
        cmap="inferno",
    )
        
    # Controls legend at bottom (using figure coordinates so it stays outside plot)
    controls_text = (
    "Control Instructions:\n"
    "Space     : Play/Pause\n"
    f"  Up        : +{time_10:.4f} time units\n"
    f"  Down      : -{time_10:.4f} time units\n"
    f"  A         : +{time_100:.4f} time units\n"
    f"  D         : -{time_100:.4f} time units")
    
    ax.set_title(controls_text, fontsize=10, family="monospace", color="white", bbox=dict(facecolor="black", alpha=0.65, boxstyle="round,pad=0.5"))
    cbar = fig.colorbar(img, ax=ax, pad=0.03)
    cbar.set_label("Temperature", rotation=270, labelpad=15)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

    # ─── Animation state ─────────────────────────────────────────────────────
    current_frame = 0
    running = False          # Start paused → shows t=0 clearly

    def update(frame):
        nonlocal current_frame, running   # ← This line fixes the UnboundLocalError

        # Check if we're in export mode (set temporarily during save)
        is_export = getattr(update, 'is_export', False)

        if is_export:
            # During GIF save → force exact frame
            display_frame = frame
        else:
            # Interactive mode → only advance if playing
            if running:
                if frame > current_frame:  # prevent going backwards during normal play
                    current_frame = min(n_frames - 1, current_frame + 1)
            display_frame = current_frame

        img.set_data(snapshots[display_frame])
        time_text.set_text(f"t = {times[display_frame]:.3f}")
        return img, time_text

    # ─── Keyboard controls ───────────────────────────────────────────────────
    def on_key(event):
        nonlocal current_frame, running

        key = event.key.lower()

        if key == " ":
            running = not running
        elif key == "up":
            running = False
            current_frame = min(n_frames - 1, current_frame + 10)
        elif key == "down":
            running = False
            current_frame = max(0, current_frame - 10)
        elif key == "a":
            running = False
            current_frame = max(0, current_frame - 100)
        elif key == "d":
            running = False
            current_frame = min(n_frames - 1, current_frame + 100)
        else:
            return

        # Refresh display
        img.set_data(snapshots[current_frame])
        time_text.set_text(f"t = {times[current_frame]:.3f}")
        event.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ─── Animation object ────────────────────────────────────────────────────
    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=50,
        blit=True,
        repeat=False
    )

    # ─── Save GIF (always full playback) ─────────────────────────────────────
    if cfg.output.save_gif:
        gif_path = getattr(cfg.output, "gif_path", "output/heat_diffusion.gif")
        print(f"Saving full animation as GIF → {gif_path}")

        # Tell update we're exporting → it will use the frame index directly
        update.is_export = True
        try:
            anim.save(
                gif_path,
                writer='pillow',
                fps=15,
                dpi=110,
                progress_callback=lambda i, n: print(f"  frame {i+1}/{n}", end='\r')
            )
            print(f"\nGIF saved successfully: {gif_path}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")
            print("→ Ensure Pillow is installed: pip install pillow")
        finally:
            if hasattr(update, 'is_export'):
                del update.is_export

    # ─── Show interactive window ─────────────────────────────────────────────
    plt.show()