# Embodied Field Intelligence (EFI)

A CA-based framework for embodied artificial intelligence using cellular automata, chemotaxis fields, and schema learning.

## Overview

EFI explores the use of cellular automata (CA) as a substrate for real-time, distributed intelligence in embodied agents. The framework combines:

- **Chemotaxis Fields**: Diffusion-based scent trails for navigation
- **Memory Systems**: Visit trails and novelty detection
- **Schema Learning**: Local Oja/BCM/slowness-based prototype learning
- **CA-Native Processing**: All computations use local field operations

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Optional Dependencies

For Gymnasium integration:
```bash
pip install -e ".[gym]"
```

For video generation:
```bash
pip install -e ".[viz]"
```

## Quick Start

### Run a Demo

```bash
# Basic demo with 5 episodes
python cli.py demo --episodes 5

# Demo with live visualization
python cli.py demo --episodes 5 --render live

# Save video of last episode
python cli.py demo --episodes 5 --save-video demo.mp4
```

### Interactive Viewer

Run an episode with an interactive viewer that allows you to:
- Play/pause animation
- Step forward/backward through frames
- Adjust playback speed
- Scrub through frames with a slider
- View all CA fields in real-time

```bash
# Launch interactive viewer
python cli.py interactive

# Launch with auto-play
python cli.py interactive --auto-play

# Configure environment
python cli.py interactive --H 20 --W 20 --max-steps 150
```

**Note**: The interactive viewer automatically detects your environment:
- **With display**: Opens matplotlib interactive window
- **Headless/SSH**: Generates an HTML file you can open in any browser

The HTML viewer provides the same controls as the matplotlib version and works on any system. After running, it will output a path like:
```
HTML viewer saved to: runs/interactive_20250816-170917.html
```

Open this file in your browser to interact with the episode.

### Run Evaluation

```bash
# Evaluate over 50 episodes with 3 seeds
python cli.py eval --episodes 50 --seeds 3 --out results/eval1

# Evaluate with specific ablations
python cli.py eval --episodes 50 --novelty 0 --schema 1 --out results/no_novelty
```

### Run Ablation Suite

```bash
# Run full ablation suite
python cli.py suite --episodes 20 --seeds 5 --out results/ablations
```

### Register Gymnasium Environment

```bash
# Register environment for use with RL libraries
python cli.py gym-register
```

Then use in your code:
```python
import gymnasium as gym
env = gym.make("CAForage-v0")
```

## Project Structure

```
embodied_field_intelligence/
├── efi/                    # Main package
│   ├── configs/           # Configuration dataclasses
│   ├── core/              # Core utilities (diffusion, fields)
│   ├── agents/            # Agent implementations
│   ├── envs/              # Environment and Gym wrapper
│   ├── evaluation/        # Experiment runners and metrics
│   └── visualization/     # Plotting and video generation
├── tests/                 # Unit tests
├── scripts/               # Public analysis and demo utilities
├── docs/                  # Documentation and reports
├── exports/               # GIF and video exports
├── results/               # Experiment results
├── runs/                  # Run artifacts and logs
└── cli.py                # Command-line interface
```

## Key Components

### Chemotaxis Agent

The `ChemotaxisAgentCA` maintains multiple CA fields:
- **GA/GB**: Attractive scent fields for targets
- **Vtrail**: Repulsive visit trail for exploration
- **Novel**: Novelty field based on prediction error
- **Corner Hazard**: Static hazard map for navigation safety

### Schema Field

The `SchemaField` learns local prototypes using:
- **Oja normalization**: Stable Hebbian learning
- **BCM threshold**: Competitive dynamics
- **Slowness penalty**: Temporal stability
- **Diffusion**: Smooth activation maps

### Ablations

Test contributions of different components:
- `--trail 0`: Disable visit trail
- `--novelty 0`: Disable novelty detection
- `--corner 0`: Disable corner hazard
- `--schema 0`: Disable schema learning

## Configuration

### Environment Parameters

- `--H`, `--W`: Grid dimensions (default: 17x17)
- `--win`: Observation window size (default: 5)
- `--p-wall`: Wall generation probability (default: 0.12)
- `--nA`, `--nB`: Number of targets (default: 3 each)
- `--max-steps`: Episode length (default: 200)

### Agent Parameters

- `--seed-strength`: Scent injection strength (default: 0.6)
- `--scent-diff`: Scent diffusion rate (default: 0.14)
- `--v-decay`: Visit trail decay rate (default: 0.03)
- `--anti-stuck-temp`: Temperature for stuck recovery (default: 0.6)

### Schema Parameters

- `--schema-tile`: Tile size for prototypes (default: 5)
- `--schema-K`: Prototypes per tile (default: 4)
- `--schema-eta`: Learning rate (default: 0.03)
- `--schema-slowness`: Slowness penalty (default: 0.02)

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=efi

# Run specific test file
pytest tests/test_diffusion.py
```

## Experiments

### Running Custom Experiments

```python
from efi.configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from efi.evaluation import run_experiment

# Configure experiment
env_cfg = EnvConfig(H=20, W=20, n_targets_A=5)
agent_cfg = AgentConfig(seed_strength=0.8)
schema_cfg = SchemaConfig(K=6, eta=0.05)
ablate = Ablations(trail=1, novelty=1, corner=1, schema=1)

# Run experiment
results = run_experiment(
    env_cfg, agent_cfg, schema_cfg, ablate,
    episodes=100, seeds=10
)

print(f"Mean return: {results.mean_return:.3f} ± {results.std_return:.3f}")
```

### Visualization

```python
from efi.visualization import plot_experiment_results

# Plot results
fig = plot_experiment_results(results, save_path="results.png")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{efi2024,
  title={Embodied Field Intelligence},
  author={Joshua Winters},
  year={2025},
  url={https://github.com/jbwinters/embodied-field-intelligence}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments

This project builds on research in:
- Cellular automata and self-organization
- Reservoir computing
- Chemotaxis and stigmergy
- Neural cellular automata (NCA)
- Artificial General Intelligence (AGI)
