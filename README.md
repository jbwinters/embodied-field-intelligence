# Embodied Field Intelligence (EFI)

[![CI](https://github.com/jbwinters/embodied-field-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/jbwinters/embodied-field-intelligence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

A framework for embodied artificial intelligence from local field dynamics. The original foraging controller uses a 5×5 window, internal belief fields, and a linearly-solvable-MDP value recursion with zero training episodes. Newer pilots learn motion and contact consequences from online experience.

**Start here:** [Watch the demos](docs/README.md#watch-a-demo) ·
[Run it locally](#quick-start) · [Read the evidence](docs/README.md#read-the-research).
The recorded HTML players work offline without Python or a GPU. On GitHub,
download the HTML file and open it in a browser; a repository link shows its source.

![A 35×35 foraging run: the agent collects every appetitive target while threading between twenty aversive ones](docs/assets/images/grid_demo.gif)

*35×35, 12 appetitive targets (green), 20 aversive ones (magenta), zero training episodes. The agent maps mines from single glances and plans around them as path costs: it collects **all 12 A while touching exactly one B** — the one it needed to taste to learn the sign. Blue trace = recent path. Reproduce: `python scripts/make_grid_demo.py`.*

### Watch it re-value the world

Every internal quantity is inspectable live: `python cli.py interactive` writes `runs/interactive_latest.html` — belief fields p(A)/p(B), the value field V, state costs q, the information-gain reward, and the policy distribution π ∝ exp(V/λ) as arrows, with telemetry strips (reward, λ, value residual, valences, affect) below and a synchronized cell probe on hover.

The episode below runs that viewer through the revaluation experiment: at step 300 the rewards swap, and everything the agent liked becomes aversive and vice versa. Because values are recomputed from beliefs every tick (not memorized), a few pickups flip the learned valences and the whole policy reverses — watch the VALENCES strip cross and the reward spikes resume on the other side of the swap line:

![Policy reversal in the episode viewer: rewards swap mid-episode, valences cross, and the agent switches targets](docs/assets/images/viewer_swap_demo.gif)

*25×25, regrowing targets, reward swap at step 300 (`python scripts/make_swap_demo.py`). This is the revaluation experiment from the paper (adaptation lag 5 steps vs 69–83 for trained baselines), playing live.*

### Watch experience change an approach

The new contact pilot learns how an object responds to contact, then uses
joint body/object predictions to approach it from a useful direction.
Across 40 held-out seeds, acquired evidence plus online learning achieves
**93.65% / 92.40%** goal collection in two rearrangements, versus
**66.46% / 65.83%** with empty evidence. Acquisition costs 80 real transitions
per model, including 16 contact attempts. This is a bounded, opt-in milestone;
the [full report](docs/INTERACTION_LEARNING.md) includes negative controls
and the supplied-versus-learned boundary.

![Learned contact consequences, shown in the original EFI episode viewer](docs/assets/images/interaction_viewer.gif)

*The original viewer shows actual fields and five-action probabilities.
[Open the replay](docs/assets/interactive/interaction.html), or generate it
with `python cli.py interaction`. Two source contacts and the first selected
target attempts are shown; omitted source interventions are labeled.*

For an easier view of the learning process, open the
[180-move continuous example](docs/assets/interactive/interaction_long.html).
One agent starts with empty memory, encounters obstacles and two changes
in contact response, and keeps learning without resets. The original viewer
now includes a legend, the chosen move, sensing boundaries, feedback, and
chapter jumps. Reproduce with `python cli.py contact-demo`.

## Overview

The proposed next architecture is described in
[A field architecture for accumulating online intelligence](docs/ONLINE_INTELLIGENCE_DESIGN.md).
It develops learned action consequences, contextual memory, and composition
under explicit CPU, memory, and locality budgets. Its first contact pilot
is documented in the [interaction-learning report](docs/INTERACTION_LEARNING.md).
Earlier demonstrated capabilities remain documented in the
[predictive control](docs/PREDICTIVE_CONTROL.md) and
[transfer](docs/PREDICTIVE_TRANSFER.md) reports.

EFI explores the use of cellular automata (CA) as a substrate for real-time, distributed intelligence in embodied agents. Local sensing seeds belief and memory fields; bounded spatial updates propagate predictions and action values. The original chemotaxis controller combines attraction and repulsion, the default foraging controller adds local value relaxation, and the newer pilots add learned motion and contact predictions. The framework combines:

- **Chemotaxis Fields**: Diffusion-based scent trails for navigation
- **Memory Systems**: Visit trails and novelty detection
- **Schema Learning**: Local Oja/BCM/slowness-based prototype learning
- **Affect & Membranes**: Nociception-driven safety fields and learning gates
- **CA-Native Processing**: Computations built from local field operations

**Results at a glance** (17×17 ForageWorld, identical seeds, 200 episodes/agent; normalized = (X − random)/(oracle − random); reproduce with `python scripts/make_baseline_table.py`):

| Agent | Mean return | Success | Normalized score | Training episodes |
|---|---|---|---|---|
| Random walk | −2.795 ± 0.909 | 6.0% | 0.00 | 0 |
| Greedy-visible | −1.386 ± 0.797 | 22.5% | 0.50 | 0 |
| Tabular Q | −2.841 ± 1.546 | 0.5% | −0.02 | 2000 |
| **EFI (this repo)** | **−0.087 ± 0.286** | **96.5%** | **0.97** | **0** |
| A* oracle (ceiling) | −0.000 ± 0.000 | 100.0% | 1.00 | 0 |

EFI reaches 97% of a full-observability oracle **with zero training episodes**, from a 5×5 window and internal field dynamics alone. The control law is a linearly-solvable-MDP value recursion computed as a local field operation (see [docs/THEORY.md](docs/THEORY.md)); success no longer degrades with grid size (98.9% at both 15×15 and 30×30). Older results: [experiment report](docs/EXPERIMENT_REPORT.md), [research site](https://jbwinters.github.io/embodied-field-intelligence/), [roadmap](docs/ROADMAP.md), [paper draft](paper/efi_paper.tex).

## Installation

Requires Python 3.10+. Run these commands from the repository root, preferably
inside a virtual environment. NumPy runs the agents on the CPU; no GPU is required.

### Basic Installation

```bash
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

### Watch continuous learning

After installation:

```bash
python cli.py contact-demo --seed 6 --max-steps 180 --out runs/contact-demo
```

Open `runs/contact-demo/episode.html` in a browser. One agent learns through
180 moves with obstacles and two changes in contact response. Use the chapter
buttons to jump to moves 60 and 120, then step through the feedback.
This is an illustration of the two-step contact learner, not a benchmark or
a demonstration of long-horizon reasoning. The [report](docs/INTERACTION_LEARNING.md)
explains the experiment and its limits.

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
- Inspect recorded CA fields

```bash
# Write an offline HTML recording of a foraging episode
python cli.py interactive

# Optional matplotlib desktop window with auto-play
python cli.py interactive --window --auto-play

# Configure environment
python cli.py interactive --H 20 --W 20 --max-steps 150
```

The default always writes HTML, including on desktop and headless systems.
Open `runs/interactive_latest.html` in your browser. `--window` explicitly
requests the matplotlib viewer instead. HTML playback replays recorded
observations, fields, and decisions; it does not run the agent in the browser.

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
│   ├── core/              # Core utilities (diffusion, fields, affect, membranes)
│   ├── agents/            # Agent implementations
│   ├── envs/              # Environment and Gym wrapper
│   ├── evaluation/        # Experiment runners and metrics
│   ├── visualization/     # Plotting and video generation
│   └── cli.py             # Command-line interface (`efi` console script)
├── tests/                 # Unit tests
├── scripts/               # Analysis and demo utilities
├── docs/                  # Documentation, reports, and research site
├── paper/                 # LaTeX paper draft
└── cli.py                 # Thin shim so `python cli.py ...` works from the repo root
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
- `--nA`, `--nB`: Number of targets (defaults: 2 A, 4 B)
- `--max-steps`: Episode length (default: 200)

### Foraging Controller Parameters

The CLI defaults to `--controller field --control-mode lmdp`. Use
`--controller chemotaxis` for the original scent controller. Scent parameters
apply to that controller and the field controller's legacy scent path.

- `--lam`: Value-control temperature (default: 0.02)
- `--z-sweeps`: Value updates per tick (default: 3)
- `--seed-strength`: Scent injection strength (default: 1.0)
- `--scent-diff`: Scent diffusion rate (default: 0.25)
- `--v-decay`: Trail decay rate (default: 0.02)
- `--anti-stuck-temp`: Temperature for stuck recovery (default: 0.8)

The contact and motion experiments have their own bounded configurations;
run `python cli.py contact-demo --help` or the relevant command's `--help`.

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

### Learned anticipation of moving hazards

An opt-in egocentric controller now learns local hazard motion and uses
four-step forecasts to choose movements or waiting. In a paired crossing
experiment it achieves 95.25% success during acquisition, 99.25% after
transfer to larger corridors, and 89.75% after a motion-rule reversal.
Static and unlearned forecast controls use the same sensing and planning
budgets. Existing foraging defaults are unchanged.

```bash
python cli.py crossing --seeds 20 --episodes 20 --seed 1000 --out runs/predictive-crossing
```

Open `runs/predictive-crossing/episode.html` to inspect the forecasts and
the actual move/wait probabilities. See the [protocol, ablations, regression
checks, and limitations](docs/PREDICTIVE_CONTROL.md).

### Reusing motion across tasks

An opt-in relational motion model reuses hazard-avoidance experience to
intercept moving rewards in rooms. On the task combining obstacles and a
moving hazard, frozen transfer achieves 81.25% success, versus 49.58% with
an empty frozen model and 48.75% with exact-context reuse. Continued learning
reaches 87.50%; a model learning from scratch reaches 87.08%. These are
20-seed results, with 240 trials per task and contender.

```bash
python cli.py transfer --seeds 20 --episodes 12 --acquisition 20 --seed 5000 \
  --out runs/predictive-transfer
```

The [report](docs/PREDICTIVE_TRANSFER.md) covers all four tasks, controls,
uncertainty, and limits. Open `runs/predictive-transfer/episode.html` for the
recorded target and hazard forecasts. The original 212 evaluation episodes
and all 4,800 previous crossing trials retain identical recorded results.

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
    episodes=100, seeds=10, use_controller=True
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
@software{efi2025,
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
