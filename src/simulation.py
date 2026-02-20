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

    Controls:
        Space: Play/Pause
        Left: Step one frame backwards
        Right: Step once frame forwards
        Down: Step ten frames downwards
        Up: Step ten frames forwards
        A: Step 100 frames backwards
        D: Step 100 frames forwards
    """
    result = sv.run(cfg)           # ← runs the solver
    snapshots = result.snapshots
    times = result.times
    print("times[0], times[-1]:", times[0], times[-1])
    print("snap 0 min/max:", snapshots[0].min(), snapshots[0].max())
    print("snap last min/max:", snapshots[-1].min(), snapshots[-1].max())

    Lx = result.metadata["Lx"]
    Ly = result.metadata["Ly"]

    n_frames = len(snapshots)
    # nonlocal update(), on_key(event)


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

    controls_text = ("Controls:\n",
                     " Space: Play/Pause\n",
                     " Left/Right: ±1 frame\n",
                     " Up/Down: ±10 frame\n",
                     " A/D: ±100 frame\n",
                     )

    ax.text(
        0.02, 0.02,
        controls_text,
        transform=ax.transAxes,
        fontsize=8,
        color="black",
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.4, edgecolor="none", pad=3),
    )

    # ----- Animation State --------

    current_frame = 0
    running = True

    def update(_):
        nonlocal current_frame, running

        if running:
            # Advance one frame, but stop at the last frame (no wrap)
            if current_frame < n_frames - 1:
                current_frame += 1
        img.set_data(snapshots[current_frame])
        time_text.set_text(f"t = {times[current_frame]:.3f}")
        return img, time_text
    
    # ----- Keyboard Control Handler ----------
    def on_key(event):
        nonlocal current_frame, running

        match event.key:
            case " ":
                running = not running
            case "up":
                running = False
                current_frame = min(n_frames - 1, current_frame + 10)
            case "down":
                running = False
                current_frame = max(0, current_frame - 10)
            case "left":
                running = False
                if current_frame > 0:
                    current_frame -= 1
            case "right":
                running = False
                if current_frame < n_frames - 1:
                    current_frame += 1
            case "a":
                running = False
                current_frame = max(0, current_frame - 100)
            case "d":
                running = False
                current_frame = min(n_frames - 1, current_frame + 100)
            case _:
                return


        img.set_data(snapshots[current_frame])
        time_text.set_text(f"t = {times[current_frame]:.3f}")
        event.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    anim = anmtion(fig, update, 
                   frames=n_frames, 
                   interval=50, 
                   blit=True,
                   )
    
    
    # ─── Save as GIF (add this block) ───────────────────────────────────────
    if cfg.output.save_gif:  # we'll add this flag to config later
        # gif_path = cfg.output.gif_path if hasattr(cfg.output, 'gif_path') else "output/heat_diffusion.gif"
        gif_path = getattr(cfg.output, "gif_path", "out/heat_diffusion.gif")

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