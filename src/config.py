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

    csv_path: str = "output/grid.csv"
    image_path: str = "output/grid.png"
    runtime_path: str = "output/runtime.json"

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
            
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save_manifest(self) -> None:
        if not self.output.save_runtime:
            return
        with open(self.output.runtime_path, 'w', encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

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

    
