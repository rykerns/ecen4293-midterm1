# holds the load and validate configuration functions (probably in JSON)

class PlateConfig:
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
    
class SolverConfig:
    #Solver parameters
    alpha: float = 0.01  # thermal diffusivity
    dt: float = 0.01     # time step size
    t_end: float = 1.0    #  total simulation time

    """ helper function to compute number of time steps """

class HotspotConfig:
    """
    Circular hotspot parameters, could be a list of hotspots if we want to support multiple hotspots.
    center: (x,y) coordinates of the center of the hotspot
    radius: radius of the hotspot
    temp: temperature of the hotspot
    """

    center: tuple[float, float] = (0.5, 0.5) # center of the hotspot in (x,y) coordinates
    radius: float = 0.1
    temp: float = 100.0

class BoundaryConfig:
    #Boundary condition parameters

    """
    BCs for each edge of the plate, could be a string like "Dirichlet" or "Neumann" or something more complex like a function that takes in the grid and returns the BC values. 
    For simplicity, we can start with just strings and then expand if needed.

    """

class OutputConfig:
    # Output parameters
    save_csv: bool = True
    save_image: bool = True
    save_runtime: bool = True

    csv_path: str = "output/grid.csv"
    image_path: str = "output/grid.png"
    runtime_path: str = "output/runtime.json"

class TopLevelConfig:
    plate: PlateConfig = PlateConfig()
    solver: SolverConfig = SolverConfig()
    boundary: BoundaryConfig = BoundaryConfig()
    output: OutputConfig = OutputConfig()

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

    
