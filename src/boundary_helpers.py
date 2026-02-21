# hold the functions for applying boundary conditions to the PDE solver

"""
Py-pde needs dict entries for each side of the plate that describe the boundary entry points for the max/min of the top/bottom and left/right
Order of operations:
1) Define supported boundary types + conventions
   - Dirichlet: fixed temperature T = value
   - Neumann: fixed normal derivative dT/dn = value (0 = insulated)
   - Side mapping: left/right -> x-min/x-max, bottom/top -> y-min/y-max
2) Validate boundary inputs
   - Ensure each side has a known type
   - Ensure each side value is numeric

3) Convert side-based BCs -> axis-based BCs as dict objects that py-pde can use
   - Build x-axis BC from (left, right)
   - Build y-axis BC from (bottom, top)
4) Map normalized BCs to the solvers API format

"""

from config import TopLevelConfig, BoundarySide, BoundaryConfig
from typing import Dict, Any

# function to make sure any side is valid
def _normalize_type(bc_type: str) -> str:
    if not isinstance(bc_type, str): #"if bc_type not str" was not interpreted correctly
        raise TypeError(f"Boundary: boundary condition must be a string, got {type(bc_type)}") 
    
    t = bc_type.strip().lower() #lowercase and remove whitespace

    if t in ("dirichlet", "d"): #can handle shorthand types too
        return "dirichlet"
    if t in ("Neumann", "neumann", "n"):
        return "neumann"
    
    raise ValueError(f"Boundary: unknown boundary condition, expected 'dirichlet' or 'neumann'")

# For each side we check: 1) if it has a valid type (via _normalize_type) and 2) has a value (float)
def _normalize_side(side: BoundarySide, side_name: str) -> BoundarySide:
    bc_type = _normalize_type(side.type)

    try: #checks if the value given is valid (must at least be a numeric value)
        val = float(side.value)
    except Exception as e:
        raise TypeError(f"Boundary: the boundary '{side_name}' has a non-numeric value: {side.value}") from e

    return BoundarySide(type=bc_type, value=val) # normalized side as (type, value)



def _side_to_pde_entry(side: BoundarySide) -> Dict[str, float]:
    # With the normalized BC's we convert it to the py-pde boundary entry, and make it into a dict[str,float]
    # py-pde takes in dict objects with the terms "value" and "derivative" for their respective BCs

    if side.type == "dirichlet":
        return {"value": side.value} 
    if side.type == "neumann":
        return {"derivative": side.value}
    
def make_bc(cfg: TopLevelConfig) -> Dict[str, Any]:
    # calls the boundary specifications
    bc = cfg.boundary

    # normalize them all
    left = _normalize_side(bc.left, "left")
    right = _normalize_side(bc.right, "right")
    bottom = _normalize_side(bc.bottom, "bottom")
    top = _normalize_side(bc.top, "top")

    #define the dict entry for the boundary conditions
    return {
        "x": {
            "lower": _side_to_pde_entry(left),
            "upper": _side_to_pde_entry(right),
        },
        "y": {
            "lower": _side_to_pde_entry(bottom),
            "upper": _side_to_pde_entry(top),
        },
    }

