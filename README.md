# Heat Diffusion Sandbox

**ECEN 4293 – Applied Numerical Methods for Python**  
**Midterm Project**  
**Team:** Ricardo Landeros Aranda, Ryan Kerns, Daniel Dubon  
**Oklahoma State University – School of Electrical and Computer Engineering**  
**Stillwater, OK**

## Abstract

This project implements a configurable **2D heat diffusion sandbox** that simulates temperature evolution on a rectangular plate using the explicit finite-difference method (FTCS scheme). The solver supports Dirichlet and Neumann boundary conditions, multiple circular hotspots, stability-enforced time stepping, and interactive animated visualization with GIF export. Eight preset scenarios demonstrate a range of physical behaviors, from classic radial cooling to insulated energy-conserving diffusion.

The software is modular, lightweight, and runs entirely in Python with NumPy and Matplotlib — no external PDE libraries required.

## Goal

Build an interactive program that simulates 2D heat diffusion on a rectangular plane and visualizes the temperature field over time. Users can configure:
- Plate size and grid resolution
- Thermal diffusivity (α) and time step (Δt)
- Initial conditions (ambient temperature + hotspots)
- Boundary conditions (Dirichlet or Neumann per edge)
- Output options (CSV, PNG, GIF animation)

The simulation uses the explicit finite-difference method for the 2D heat equation, with automatic CFL stability checking.

## Testable Functional Requirements (Completed)

- [x] Specify grid resolution and physical size  
- [x] Specify thermal diffusivity and time step  
- [x] Support initial conditions (temperature, location of hot-spot(s), radius)  
- [x] Implement the finite-difference method for the 2D heat equation  
- [x] Support boundary conditions (Dirichlet and Neumann)  
- [x] Display a live heatmap of the temperature field (with color bar)  
- [x] Save final heatmap data to CSV file  
- [x] Save at least one plot of the final heatmap as an image (PNG)  
- [x] Include demo scenario presets for the user (8 provided)  
- [x] Show runtime stats (min/max/mean temperature during simulation)

## Features

- **Modular design** — separate layers for configuration (`config.py`), numerical solver (`solver.py`), and visualization (`simulation.py`)
- **JSON presets** — easily load and share simulation scenarios
- **Interactive animation** — real-time playback with keyboard controls (play/pause, step forward/backward by 1/10/100 frames)
- **GIF export** — save full animation for reports/presentations
- **Stability enforcement** — automatic CFL check prevents unstable runs
- **Debug & validation** — clear warnings if initial conditions are missing or invalid

## Project Structure
src/
├── config.py           # Central configuration, JSON preset loading, validation, main entry point
├── solver.py           # Explicit FTCS solver, boundary handling, stability check
├── simulation.py       # Matplotlib animation with keyboard controls & GIF export
├── boundary_helpers.py # Boundary configurations
├── presets/            # Example JSON scenario files (8 presets)
└── output/             # Generated CSV, PNG, GIF, runtime JSON