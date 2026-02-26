# virtual.reality

A modular Python package for rendering virtual fly stimuli in neuroscience experiments and calibrating projector-camera systems for accurate stimulus delivery.

## Overview

This system renders a 3D fly model (GLB format) in a circular arena and projects the output through a calibrated projector warp map. It is used to present realistic visual stimuli of a walking fly to real flies in a laboratory setup.

### Key Features

- **3D GLB fly rendering** with OpenGL shaders, Phong lighting, and multiple projection modes (perspective, equidistant fisheye, equirectangular)
- **Projector-camera warp correction** using calibrated mapx/mapy lookup textures
- **Structured-light calibration** pipeline (Gray code + sine fringe hybrid)
- **Fisheye and pinhole camera calibration** with robust outlier filtering
- **Autonomous fly AI** with run/pause state machine and edge avoidance
- **2D minimap overlay** showing fly position, camera FOV cone, and movement trail
- **Dear ImGui GUI** for live parameter tuning (optional)

## Installation

```bash
# Basic installation
pip install -e .

# With camera drivers
pip install -e ".[alvium]"   # Allied Vision Alvium
pip install -e ".[flir]"     # FLIR/Spinnaker via RotPy

# With GUI
pip install -e ".[gui]"

# Development (includes pytest, ruff, mypy)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## CLI Entry Points

```bash
vr-fly3d      # 3D GLB fly stimulus
vr-fly2d      # 2D sprite fly stimulus
vr-warp-test  # Warp circle calibration test
vr-calibrate  # Calibration pipeline
vr-gui        # Dear ImGui GUI
```

## Package Structure

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

## Configuration

Create a YAML config file or use defaults:

```python
from virtual_reality.config.schema import VirtualRealityConfig
from virtual_reality.config.loader import load_config, save_config

# Load with defaults
config = VirtualRealityConfig()

# Load from file (merges with defaults)
config = load_config("my_config.yaml")

# Save current config
save_config(config, "my_config.yaml")
```

Key configuration parameters:
- `fly_model.phys_length_mm`: Target physical fly size (default: 3.0 mm)
- `camera.projection`: Projection mode (`"perspective"`, `"equirect"`, `"equidistant"`)
- `camera.fov_x_deg`: Camera horizontal FOV (default: 200.0 degrees)
- `arena.radius_mm`: Circular arena radius (default: 40.0 mm)
- `autonomous.enabled`: Toggle autonomous vs keyboard fly control

## Testing

```bash
pytest tests/
```

Tests are organized by module. GPU-dependent tests are marked with `@pytest.mark.gpu` and hardware tests with `@pytest.mark.hardware` (both skipped by default).

## Dependencies

- **Core**: numpy, opencv-python, pygame, PyOpenGL, pygltflib, PyYAML
- **Optional**: screeninfo (monitor detection), harvesters (Alvium cameras), rotpy (FLIR cameras), imgui[pygame] (GUI)

## License

MIT
