# wraps the PDE solver and the step/run functions
# Import PDE solver and step/run functions

import solver as sv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as anmtion
import config as con

def run_and_animate(cfg):
    """
    Run the solver with given config and show the animation.
    This is the function that config.py will call.
    """
    result = sv.run(cfg)           # ← runs the solver

    snapshots = result.snapshots
    times = result.times
    Lx = result.metadata["Lx"]
    Ly = result.metadata["Ly"]

    fig, ax = plt.subplots()
    img = ax.imshow(
        snapshots[0],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
        cmap="inferno",
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Temperature")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

    def update(frame):
        img.set_data(snapshots[frame])
        time_text.set_text(f"t = {times[frame]:.3f}")
        return img, time_text

    anim = anmtion(fig, update, frames=len(snapshots), interval=50, blit=True)
    
    # ─── Save as GIF (add this block) ───────────────────────────────────────
    if cfg.output.save_gif:  # we'll add this flag to config later
        gif_path = cfg.output.gif_path if hasattr(cfg.output, 'gif_path') else "output/heat_diffusion.gif"
        
        print(f"Saving animation as GIF → {gif_path}")
        try:
            anim.save(
                gif_path,
                writer='pillow',           # uses Pillow (built-in)
                fps=15,                    # adjust for smoother/faster GIF
                dpi=100,                   # balance quality vs file size
                progress_callback=lambda i, n: print(f"  frame {i+1}/{n}", end='\r')
            )
            print(f"\nGIF saved successfully: {gif_path}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")
            print("→ Make sure Pillow is installed: pip install pillow")

    # ─── Show live (keep your original plt.show()) ───────────────────────────
    plt.show()