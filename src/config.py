# holds the load and validate configuration functions (probably in JSON)

from __future__ import annotations
from dataclasses import field, asdict, dataclass
from typing import Literal, Dict, Any, Tuple
import json

BCType = Literal["dirichlet", "neumann"]

@dataclass
class PlateConfig: #done
    #Physical size parameters
    Lx: float = 1.0
    Ly: float = 1.0
    #Grid parameters
    Nx: int = 100
    Ny: int = 100

    # helper functions to compute grid spacing
    def dx(self) -> float:
        return self.Lx / (self.Nx - 1)
    
    def dy(self) -> float:
        return self.Ly / (self.Ny - 1)

@dataclass
class SolverConfig: #done
    #Solver parameters
    alpha: float = 0.01  # thermal diffusivity
    dt: float = 0.001     # time step size
    t_end: float = 1.0    #  total simulation time

    """ helper function to compute number of time steps """
    def steps(self) -> int:
        return max(1, int(round(self.t_end / self.dt)))

@dataclass
class HotspotConfig: #done
    """
    Circular hotspot parameters, could be a list of hotspots if we want to support multiple hotspots.
    center: (x,y) coordinates of the center of the hotspot
    radius: radius of the hotspot
    temp: temperature of the hotspot
    """

    center: tuple[float, float] = (0.5, 0.5) # center of the hotspot in (x,y) coordinates
    radius: float = 0.1
    temp: float = 100.0

@dataclass
class BoundarySide: # done
    """
    One side boundary condition:
    -derichlet: T=value
    -neumann: dT/dn = value (insulated = 0)
    """
    type: BCType = "dirichlet"
    value: float = 0.0

@dataclass
class InitialConditionConfig:
    ambient_temperature: float = 0.0
    hotspots: Tuple[HotspotConfig, ...] = field(default_factory=tuple)

@dataclass
class BoundaryConfig: #done
    #Boundary condition parameters

    left: BoundarySide = field(default_factory=lambda: BoundarySide("dirichlet", 0.0))
    right: BoundarySide = field(default_factory=lambda: BoundarySide("dirichlet", 0.0))
    bottom: BoundarySide = field(default_factory=lambda: BoundarySide("dirichlet", 0.0))
    top: BoundarySide = field(default_factory=lambda: BoundarySide("dirichlet", 0.0))

    """
    BCs for each edge of the plate, could be a string like "Dirichlet" or "Neumann" or something more complex like a function that takes in the grid and returns the BC values. 
    For simplicity, we can start with just strings and then expand if needed.

    """

@dataclass
class RuntimeConfig:
    show_live_plot: bool = True
    fps: int = 30
    stats_steps: int = 20  # stat printing interval

@dataclass
class OutputConfig:
    # Output parameters
    save_csv: bool = True
    save_image: bool = True
    save_runtime: bool = True
    save_gif: bool = True               # ← new
    csv_path: str = "output/grid.csv"
    image_path: str = "output/grid.png"
    runtime_path: str = "output/runtime.json"
    gif_path: str = "output/heat_diffusion.gif"   # ← new


@dataclass
class TopLevelConfig:
    # loads each of the classes and creates a fresh object for each
    plate: PlateConfig = field(default_factory=PlateConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    initials: InitialConditionConfig = field(default_factory=InitialConditionConfig)

    preset_name: str | None = None

    def validate(self) -> None:
        # plate dimenstions
        if self.plate.Lx <=0 or self.plate.Ly <=0:
            raise ValueError("Plate: physical size value must be non-negative")
        if self.plate.Nx < 3 or self.plate.Ny < 3:
            raise ValueError("Plate: Grid must have at least 3 points in each direction (Nx, Ny > 3)")
        
        #solver
        if self.solver.alpha <= 0:
            raise ValueError("Solver: Thermal diffusivity must be non-negative")
        if self.solver.dt <= 0:
            raise ValueError("Solver: dt must be > 0.")
        if self.solver.t_end <= 0:
            raise ValueError("Solver: t_end must be > 0.")
        
        #hotspots: check bounds and radius
        for h in self.initials.hotspots:
            x, y = h.center
            if not (0.0 <= x <= self.plate.Lx and 0.0 <= y <= self.plate.Ly):
                raise ValueError(f"Hotspot: center {h.center} is outside the plate.")
            if h.radius <= 0:
                raise ValueError("Hotspot: radius must be > 0.")
            
        # === Output paths – basic sanity === Added some basic checks for output paths and formats,
        # Takes care of the Things to consider points about output paths and formats in the validate() function.
        if self.output.save_csv and not self.output.csv_path.strip():
            raise ValueError("CSV save enabled but csv_path is empty")
        if self.output.save_image and not self.output.image_path.strip():
            raise ValueError("Image save enabled but image_path is empty")

        if self.output.save_image:
            ext = self.output.image_path.lower()[-4:]
            if ext not in (".png", ".jpg", ".jpeg"):
                print("Warning: image_path does not end with .png/.jpg/.jpeg — may fail to save")

        # === Boundary conditions – basic check ===
        for side_name, side in [
            ("left", self.boundary.left),
            ("right", self.boundary.right),
            ("bottom", self.boundary.bottom),
            ("top", self.boundary.top),
        ]:
            if side.type not in ("dirichlet", "neumann"):
                raise ValueError(f"Invalid boundary type on {side_name}: {side.type}")
            
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save_manifest(self) -> None:
        if not self.output.save_runtime:
            return
        with open(self.output.runtime_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    # NEW: Safe preset loading (closes the TODO about preset fallback)     
    @classmethod
    def from_preset(cls, preset_path: str) -> 'TopLevelConfig':
        """
        Load configuration from JSON preset with proper nested object construction.
        """
        default_cfg = cls()
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # ─── Plate ───────────────────────────────────────────────────────
            plate = PlateConfig(**data.get('plate', {}))

            # ─── Solver ──────────────────────────────────────────────────────
            solver = SolverConfig(**data.get('solver', {}))

            # ─── Boundary (nested BoundarySide objects) ──────────────────────
            boundary_data = data.get('boundary', {})
            boundary = BoundaryConfig(
                left=BoundarySide(
                    type=boundary_data.get('left', {}).get('type', 'dirichlet'),
                    value=float(boundary_data.get('left', {}).get('value', 0.0))
                ),
                right=BoundarySide(
                    type=boundary_data.get('right', {}).get('type', 'dirichlet'),
                    value=float(boundary_data.get('right', {}).get('value', 0.0))
                ),
                bottom=BoundarySide(
                    type=boundary_data.get('bottom', {}).get('type', 'dirichlet'),
                    value=float(boundary_data.get('bottom', {}).get('value', 0.0))
                ),
                top=BoundarySide(
                    type=boundary_data.get('top', {}).get('type', 'dirichlet'),
                    value=float(boundary_data.get('top', {}).get('value', 0.0))
                )
            )

            # ─── Initials (nested HotspotConfig objects) ─────────────────────
            initials_data = data.get('initials', {})
            hotspots_raw = initials_data.get('hotspots', [])
            hotspots = []
            for h_dict in hotspots_raw:
                center = h_dict.get('center', [0.5, 0.5])
                hotspots.append(HotspotConfig(
                    center=(float(center[0]), float(center[1])),
                    radius=float(h_dict.get('radius', 0.1)),
                    temp=float(h_dict.get('temp', 100.0))
                ))

            initials = InitialConditionConfig(
                ambient_temperature=float(initials_data.get('ambient_temperature', 0.0)),
                hotspots=tuple(hotspots)
            )

            # ─── Output & Runtime ────────────────────────────────────────────
            output = OutputConfig(**data.get('output', {}))
            runtime = RuntimeConfig(**data.get('runtime', {}))

            # ─── Build final config ──────────────────────────────────────────
            cfg = cls(
                plate=plate,
                solver=solver,
                boundary=boundary,
                output=output,
                runtime=runtime,
                initials=initials,
                preset_name=data.get('preset_name')
            )

            cfg.validate()
            print(f"Successfully loaded preset: {preset_path}")
            print(f"  - {len(hotspots)} hotspots loaded")
            return cfg

        except Exception as e:
            print(f"Failed to load preset '{preset_path}': {type(e).__name__}: {e}")
            print("→ Falling back to default configuration.")
            default_cfg.validate()
            return default_cfg

    # TODO: at the moment when the user supplies some preset file, it will just try to apply the overrides listed in the dataclass. 
    # Need handling for if it fails validation, either revert problem field to default, or revert all fields to default



    """
    This is where we will check that all the values are valid (e.g. positive grid size, positive time step, etc.) and raise errors if not. 
    We can also add any derived parameters here if needed (e.g. number of time steps based on t_end and dt).

    Things to consider:
    -raise values error if any of the parameters are invalid (negative grid size, negative time step, etc.)
    -compute any derived parameters (number of time steps based on t_end and dt)
    -check the boundaries of the grid ( if we have a Dirichlet BC, we need to make sure we have values for the edges of the grid)
    -check that the output paths are valid (if save_csv is True, we need to make sure csv_path is a valid path)
    -check that the output formats are valid (if save_image is True, we need to make sure image_path ends with .png or .jpg or something like that)
    -check that the hotspots are within the grid (if we have a hotspot at (x,y), we need to make sure x is between 0 and Lx; and y is between 0 and Ly)
    """
 # =============================================================================
#                           MAIN ENTRY POINT
# When running: python config.py  [ --preset path/to/preset.json ]
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import numpy as np
    import os
    # ─── Import solver & simulation ONLY here (avoids circular import) ───────
    from solver import run, SimulationResult
    from simulation import run_and_animate   # ← the new function we added

    parser = argparse.ArgumentParser(description="Heat Diffusion Sandbox")
    parser.add_argument("--preset", "-p", type=str, default=None,
                        help="Path to JSON preset file")
    parser.add_argument("--save-csv", action="store_true",
                        help="Force saving CSV even if disabled in config")
    args = parser.parse_args()

    # Load configuration
    if args.preset:
        print(f"→ Loading preset: {args.preset}")
        cfg = TopLevelConfig.from_preset(args.preset)
    else:
        print("→ Using default configuration")
        cfg = TopLevelConfig()

    # Validate
    try:
        cfg.validate()
    except ValueError as e:
        print("\nError in configuration:")
        print(str(e))
        sys.exit(1)

    # Force CSV save if requested from command line
    if args.save_csv:
        cfg.output.save_csv = True
        print("CSV saving forced via command-line flag")

    print("\nSimulation parameters:")
    print(f"  Plate: {cfg.plate.Lx:.2f} × {cfg.plate.Ly:.2f}")
    print(f"  Grid:  {cfg.plate.Nx} × {cfg.plate.Ny}")
    print(f"  α = {cfg.solver.alpha:.5f}, dt = {cfg.solver.dt:.5f}, t_end = {cfg.solver.t_end:.2f}")

    # Run simulation + show animation
    print("\nRunning simulation and animation...")
    run_and_animate(cfg)

    # After animation closes → save CSV (if enabled)
    # Note: we re-run solver very briefly only to get final_grid
    #       (not ideal, but keeps simulation.py untouched)
    if cfg.output.save_csv:
        print("Saving final grid to CSV...")
        result = run(cfg, store_every=None)   # no snapshots needed
        csv_dir = os.path.dirname(cfg.output.csv_path) or "."
        os.makedirs(csv_dir, exist_ok=True)
        np.savetxt(
            cfg.output.csv_path,
            result.final_grid,
            delimiter=",",
            fmt="%.6f",
            header=f"Final temperature field (shape {result.final_grid.shape})"
        )
        print(f"→ Saved to: {cfg.output.csv_path}")

    print("\nDone.")