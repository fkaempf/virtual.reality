# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**virtual.reality** is a modular Python package for rendering virtual fly stimuli in neuroscience experiments and calibrating projector-camera systems. It consolidates the former `virtual.fly` (24 Python scripts) and `screen.calibration` (~15 files) repositories into a single well-structured package.

## Architecture

### Package Structure

```
src/virtual_reality/
    config/           # Dataclass schemas, YAML loader, platform paths
    cameras/          # Camera Protocol + Alvium/RotPy drivers + factory
    calibration/      # Fisheye/pinhole calibration, intrinsics I/O
    mapping/          # Structured light, warp maps, pipeline, refinement
    rendering/        # GL utils, shaders, GLB loader, projections
    math_utils/       # Matrix transforms, arena geometry, lighting
    stimulus/         # Stimulus ABC + Fly3D/FlySprite/WarpCircle + controllers
    display/          # Monitor picking, surface conversion, minimap, window
    pipeline/         # Calibration pipeline orchestrator
    gui/              # Dear ImGui application + panels
    legacy/           # Archived original scripts (read-only reference)
```

### Key Concepts

- **Projector warp maps**: `mapx.npy`/`mapy.npy` files define pixel mapping from projector space to camera space
- **Structured-light**: Gray code for integer precision + sine phase for subpixel accuracy
- **Camera Protocol**: `cameras.base.Camera` defines the interface (start/grab/stop)
- **Config system**: Nested dataclasses with YAML serialization (`config.schema.VirtualRealityConfig`)
- **Stimulus lifecycle**: `setup() -> update(dt, events) -> render() -> teardown()`

## Build & Test Commands

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=virtual_reality

# Type checking
mypy src/virtual_reality/

# Linting
ruff check src/
```

## CLI Entry Points

```bash
vr-fly3d      # 3D GLB fly stimulus
vr-fly2d      # 2D sprite fly stimulus
vr-warp-test  # Warp circle calibration test
vr-calibrate  # Calibration pipeline
vr-gui        # Dear ImGui GUI
```

## Dependencies

- **Core**: numpy, opencv-python, pygame, PyOpenGL, pygltflib, PyYAML
- **Optional**: screeninfo, harvesters (Alvium), rotpy (FLIR), imgui[pygame] (GUI)
- **Dev**: pytest, pytest-cov, ruff, mypy

## Coding Standards

- Python 3.10+ (use `X | Y` union syntax, not `Optional[X]`)
- Google Python Style Guide docstrings
- Type annotations on all public APIs
- Tests written before implementation (TDD)
- Conventional commit messages: `feat(module): description`

## Test Markers

- `@pytest.mark.gpu`: Requires OpenGL context (skipped by default)
- `@pytest.mark.hardware`: Requires physical camera/projector (skipped by default)

## Legacy Files

Original scripts are archived in `src/virtual_reality/legacy/` for reference. The key decomposition:

| Legacy File | New Location |
|---|---|
| `3d_object_fly4.py` | `rendering/*`, `stimulus/fly_3d.py`, `stimulus/autonomous.py`, `display/minimap.py`, `math_utils/*` |
| `mapping_utils.py` | `mapping/structured_light.py` |
| `fisheye_KDxi.py` | `calibration/fisheye.py` |
| `mapping_pipeline.py` | `mapping/pipeline.py` |
| `CamAlvium.py` / `CamRotPy.py` | `cameras/alvium.py` / `cameras/rotpy_driver.py` |
