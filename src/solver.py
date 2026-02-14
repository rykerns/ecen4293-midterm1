# Build grid
    # CartesianGrid expects bounds as [[xmin,xmax],[ymin,ymax]] and shape as [nx, ny] or [ny,nx] depending on axes order.
    # We'll treat x as axis 0, y as axis 1 in the grid definition, then map data

# Initial scalar field: py-pde expects data aligned with grid axes.
    # Our u0 is (ny, nx) with rows=y, cols=x. Convert to (nx, ny) for ScalarField on [x,y] axes.
    # update: py-pde is not needed with the Finite difference version, but the process is still pretty much the same

# Initially planned to store the snapshots optionally, but moved away from that; yopu may see leftover parts from that


import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional, List

from config import TopLevelConfig
from boundary_helpers import make_bc


# progress callback signature:
#   progress(step_index, time_seconds, grid_2d)
ProgressCallback = Callable[[int, float, np.ndarray], None]

@dataclass
class SimulationResult:
    """
    Pure-compute return object.

    - times: time values corresponding to each stored snapshot
    - snapshots: stack of (Ny, Nx) arrays, or None if not stored
    - final_grid: final (Ny, Nx) temperature field
    - metadata: run details useful for file handlers / plotting / provenance
    """
    times: np.ndarray # (K,)
    snapshots: np.ndarray # (K, Ny, Nx) if stored
    final_grid: np.ndarray # (Ny, Nx)
    metadata: Dict[str, Any]

def _apply_bc(T: np.ndarray, bc:Dict[str, Any], dx: float, dy: float) -> None:
    """
    Apply Dirichlet/Neumann BCs in-place to the edges of T.

    T shape (Ny, Nx)
      left(x lower)-> col 0
      right(x upper)-> col -1
      bottom(y lower)-> row 0
      top(y upper)-> row -1

    Neumann values are outward normal derivatives dT/dn.
    """

    # for each side, we get the axis and the end that it is applied to
    x_lower = bc.get("x", {}).get("lower", {})
    x_upper = bc.get("x", {}).get("upper", {})

    y_lower = bc.get("y", {}).get("lower", {})
    y_upper = bc.get("y", {}).get("upper", {})

    # These are all prettu much the same:
    #   1. We choose the side we are working on by getting the dict from bc (looks like {"value", ...} or {"derivative", ...})
    #   2. Apply the chosen boundary condition based on which was chosen
    #       a. "value" we apply Dirichlet BC: T(edge) = constant; then set edge cells to that constant
    #       b. "derivative" we apply Neumann: dT/dn = g; then convert the derivative into the edge tempurature using a neighbor cell (edge = neighbor + g * spacing)

    """
    Neumann is “normal derivative fixed”:
    Left edge (normal points to -x): T[:,0] = T[:,1] + g*dx
    Right edge (normal points to +x): T[:,-1] = T[:,-2] + g*dx
    Bottom edge (normal points to -y): T[0,:] = T[1,:] + g*dy
    Top edge (normal points to +y): T[-1,:] = T[-2,:] + g*dy
    """

    if "value" in x_lower:
        T[:, 0] = float(x_lower["value"])
    elif "derivative" in x_lower:
        g = float(x_lower["derivative"])
        T[:, 0] = T[:, 1] + g * dx

    if "value" in x_upper:
        T[:, -1] = float(x_upper["value"])
    elif "derivative" in x_upper:
        g = float(x_upper["derivative"])
        T[:, -1] = T[:, -2] + g * dx

    if "value" in y_lower:
        T[0, :] = float(y_lower["value"])
    elif "derivative" in y_lower:
        g = float(y_lower["derivative"])
        T[0, :] = T[1, :] + g * dy

    if "value" in y_upper:
        T[-1, :] = float(y_upper["value"])
    elif "derivative" in y_upper:
        g = float(y_upper["derivative"])
        T[-1, :] = T[-2, :] + g * dy

 # Run will return the "SimulationResult" object which will be used to pass arguements to the interface/simulation
def run(cfg: TopLevelConfig, store_every: Optional[int] = None, progress: Optional[ProgressCallback] = None) -> SimulationResult:
    """
    Explicit FD solver for 2D heat equation:
        dT/dt = alpha (d2T/dx2 + d2T/dy2)

    BCs supported on all sides:
    - Dirichlet: T = value
    - Neumann: dT/dn = value(normal derivative)

    Grid convention:
    - T has shape (Ny, Nx)
    - axis 0 = y (bottom->top), axis 1 = x (left->right)
    """

    # load and validate the objects from TopLevelConfig
    cfg.validate()

    plate = cfg.plate
    solver = cfg.solver
    runtime = cfg.runtime
    initials = cfg.initials

    Nx, Ny = int(plate.Nx), int(plate.Ny)
    dx, dy = float(plate.dx()), float(plate.dy())
    alpha = float(solver.alpha)
    dt = float(solver.dt)
    steps = int(solver.steps())

    # Do we want to store everything within the runtime?
    if store_every is None:
        store_every = int(getattr(runtime, "store_every", 1))
    store_every = max(1, int(store_every))

    # Stability check for explicit 2D diffusion (must satisfy):
    # dt_max <= 1 / (2*alpha*(1/dx^2 + 1/dy^2))
    denom = 2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2)
    if denom > 0:
        dt_max = 1.0 / denom
        if dt > dt_max:
            raise ValueError(
                f"Solver: Unstable explicit value: dt={dt} > dt_max≈{dt_max:.6g}. Decrease dt or increase grid spacing."
            )
    
    # Build physical coords for IC construction
    xs = np.linspace(0.0, float(plate.Lx), Nx)
    ys = np.linspace(0.0, float(plate.Ly), Ny)
    X, Y = np.meshgrid(xs, ys)  # shapes (Ny, Nx)

    # --GRID OBJECT--
    #
    T = np.full((Ny, Nx), float(initials.ambient_temperature), dtype=float)

    # Apply hotspots (overwrite inside circle)
    for h in initials.hotspots:
        cx, cy = h.center
        r2 = float(h.radius) ** 2
        mask = (X - float(cx)) ** 2 + (Y - float(cy)) ** 2 <= r2
        T[mask] = float(h.temp)

    # Convert BC parts into solver dict
    # Didnt end up needing py-pde for this, but I'll still just reuse the dicts we made in boundary_helpers
    bc = make_bc(cfg)

    times: List[float] = []
    snaps: List[np.ndarray] = []

    # logging method for the solver, saves the snapshot at the time 't' every 'n' steps
    def record(step_index: int, t: float, grid: np.ndarray) -> None:
        if (step_index % store_every) == 0: #store every 'n' steps (whatever the user defines)
            times.append(float(t)) #saves the current time
            snaps.append(grid.copy()) #saves the current grid temperature profile
        if progress is not None:
            progress(step_index, float(t), grid) #returns the current state of the grid, rather than continuous snapshots
    
    # Initial grid snapshot (always exists in any preset)
    record(0, 0.0, T)

    # Need a def to apply the boundary conditions to the generated grid

    # Time stepping:
    for k in range(1, steps + 1):
        _apply_bc(T, bc, dx=dx, dy=dy) #update edges of T each step so that the FD uses the correct boundary values in each step

        Tn = T.copy() #safety so that values in T are not overwritten

        # We calculate the second derivative (finite differences)
        #   1. Txx is (partial) d^2/dx^2 using a "neighbored" set of points along the grid. Shape: (Ny, Nx-2) since it cant be computed at the left/right boundaries
        #   2. Tyy is similar only that the shape cant be computed at the top/bottom boundaries

        # To approximate Txx at index i, we need to solve:
        #   Txx(i) \approx (T_next - 2T_current + T_previous) / dx^2
        # Thus the neighboring points used are (i-1, i, i+1) which have coefficients of (1, -2, 1)
        Txx = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) / dx**2 #we apply the three neighbor rule for x in each row
        Tyy = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) / dy**2 #same but for y in eachg column

        # Then we use Eulers method to compute the explicit step change along our grid
        #   T_new = T_old + (step)*alpha* (Laplacian of T)
        #   Laplacian of T in 2D is (nabla)^2 * T = Txx + Tyy
        Tn[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * alpha * (Txx[1:-1, :] + Tyy[:, 1:-1])

        T = Tn #replace the current grid with the new step
        t = min(k * dt, float(solver.t_end)) #log the time

        record(k, t, T)

    # Final boundary condition application
    _apply_bc(T, bc, dx=dx, dy=dy)

    snapshots_arr = np.stack(snaps, axis=0)
    times_arr = np.array(times, dtype=float)

    # Final simulation object
    return SimulationResult(times=times_arr, snapshots=snapshots_arr, final_grid=T, metadata=
        {
            "Nx": Nx,
            "Ny": Ny,
            "Lx": float(plate.Lx),
            "Ly": float(plate.Ly),
            "dx": dx,
            "dy": dy,
            "alpha": alpha,
            "dt": dt,
            "t_end": float(solver.t_end),
            "steps": steps,
            "store_every": store_every,
        },
    )









