# CLI entrypoint OR JSON files OR CSV files

# ---Minor testbench to check that the solver works (can remove if needed)---

from config import TopLevelConfig, HotspotConfig
from solver import run


def main():
    cfg = TopLevelConfig()
    cfg.initials.hotspots = (HotspotConfig(center=(0.5, 0.5), radius=0.1, temp=100.0),)
    res = run(cfg, store_every=5)

    print("Done.")
    print(f"final min/max: {res.final_grid.min():.3f} / {res.final_grid.max():.3f}")
    print(f"snapshots: {res.snapshots.shape[0]}, grid: {res.final_grid.shape}")
    
    mins = res.snapshots.min(axis=(1,2))
    maxs = res.snapshots.max(axis=(1,2))

    print("t0 min/max:", mins[0], maxs[0])
    print("t_end min/max:", mins[-1], maxs[-1])
    print("max change:", (maxs - maxs[0]).min(), (maxs - maxs[0]).max())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

