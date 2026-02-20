# maybe plotly?

# Instantiate top-level config and calls solver with the selected config.
cfg = con.TopLevelConfig() ###################################################
result = sv.run(cfg)        # Returns simulation with all the info###########################

#Making a 3D array od the shape K Ny Nx
snapshots = result.snapshots        # (K (stored frames), Ny (y grid points), Nx (x grid points))
times = result.times                # (K,) Extracts the array of times tied to every snapshot
Lx = result.metadata["Lx"]          # Horizontal plane length
Ly = result.metadata["Ly"]          # Vertical plane length


# Pass parameters into 2D array image display
fig, ax = plt.subplots() # Create figure
img = ax.imshow(
    snapshots[0],
    origin="lower",
    extent=[0, Lx, 0, Ly],
    aspect="equal",
    cmap="inferno",
)
cbar = fig.colorbar(img, ax=ax) # Add color bar for reference
cbar.set_label("Temperature")   # Adds a label to the color bar

# Drawing text inside the colormap
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color="white")

# Frame generation and update
def update(frame):
    img.set_data(snapshots[frame])
    time_text.set_text(f"t = {times[frame]:.3f}")
    return img, time_text

# Display frames as an animation
anim = anmtion(fig, update, frames=len(snapshots), interval=50, blit=True)###########################
plt.show()######################################

