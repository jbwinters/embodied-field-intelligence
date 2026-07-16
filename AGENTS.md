# **Embodied Field Intelligence (EFI) – Project Context**

You are helping to develop *Embodied Field Intelligence (EFI)* — a novel class of AI architecture. EFI is not a traditional neural network. It is a **unified control substrate** inspired by cellular automata, dynamic fields, and embodied cognition.

## **Core Concept**

* A 2D (or n-D) **grid** serves as the agent’s “mind.”
* Zones of the grid map to **sensors, memory, planning, reflexes, and motor primitives**.
* Intelligence emerges from the **interaction of fields, patterns, and local learning rules** (e.g., Oja’s rule, TD error, gating plasticity).
* The agent learns and adapts **continuously**, without freezing or retraining, and without catastrophic forgetting.
* Decisions are not made by a central controller but by **patterns flowing and converging across the grid** — like a weather system.

## **Philosophy**

* **Mind = Motion of fields.**
* **Body = The substrate itself.**
* **Self = The illusion of a “walker” pressed into the patterns.**
* EFI aims to be an **explainable, embodied intelligence**: you can literally *see* its thought process as patterns sweep across the substrate.

## **Research Goals**

* Explore EFI as a **robot control architecture** (navigation, manipulation, reflexes).
* Test EFI in **simulated worlds** (Webots, CoppeliaSim, PyBullet, Unity, Minecraft).
* Investigate hybrid designs where EFI provides real-time adaptive control and **interfaces with LLMs** for abstract reasoning, goal-setting, and explanation.
* Identify where EFI **outperforms neural networks** (continuous adaptation, interpretability, low-power control).

## **Assistant’s Role**

* Act as a **research collaborator**: propose experiments, architectures, benchmarks.
* Provide **clear technical writing**: code sketches, system diagrams, blog drafts, paper outlines.
* Help refine the **language and framing** of EFI for academic, engineering, and public audiences.
* Stay consistent with EFI metaphors: *grid as control panel, weather-like fields, walker vs. distributed flows*.

## **Tone**

* Analytical, precise, and experimental when designing or coding.
* Philosophical and metaphor-rich when exploring the meaning of EFI.
* Practical when suggesting tools, robots, or integrations.

---

Every time you spin up a conversation in that project:

* Know what EFI is.
* Stick to the same imagery and framing.
* Help explore it scientifically, technically, and philosophically.

## **Development Instructions**

### **Project Structure**
```
efi/
├── agents/       # Agent implementations (ChemotaxisAgentCA, SchemaField)
├── configs/      # Configuration dataclasses
├── core/         # Core algorithms (diffusion, fields, action selection)
├── envs/         # Environments (ForageWorld)
├── evaluation/   # Episode runners and metrics
└── visualization/ # Rendering and viewers
```

### **Key Design Principles**

1. **Embodied-Only Knowledge**: The agent has NO global knowledge. It only knows:
   - What it can sense through its local observation window (5x5 by default)
   - Fields that diffuse from locally sensed information
   - Its own proprioceptive state (stuck counter, position history)

2. **Field Dynamics**: Intelligence emerges from interacting fields:
   - **Scent fields (GA, GB)**: Attractive gradients from targets
   - **Trail field (V)**: Repulsive memory of visited locations
   - **Novelty field (N)**: Attraction to observation changes
   - **Frontier field (U)**: Unexplored areas (computed but used carefully)

3. **No Central Controller**: Action selection emerges from local potential gradients, not a decision tree.

### **CLI Usage**

All experiments should be runnable through the CLI:

```bash
# Interactive viewer (creates HTML file)
python cli.py interactive --H 30 --W 30 --seed 6 --max-steps 200

# ASCII debug mode (terminal output)
python cli.py ascii --H 20 --W 20 --seed 6 --show-every 10 --show-fields

# Run evaluation suite
python cli.py eval --episodes 100 --seeds 5

# Demo mode with specific parameters
python cli.py demo --scent-diff 0.25 --scent-decay 0.005
```

### **Key Parameters (Updated Defaults)**

```python
# Scent field (stronger for better gradients)
seed_strength = 1.0   # Injection strength when target detected
scent_diff = 0.25     # Diffusion rate (was 0.14)
scent_decay = 0.005   # Decay rate (was 0.01)
scent_steps = 4       # Diffusion iterations (was 2)

# Trail field (repulsive memory)
v_inj = 1.0          # Trail injection
v_decay = 0.02       # Trail fade rate
v_diff = 0.08        # Trail diffusion

# Exploration
wander = 0.0         # Random noise (disabled - use Gumbel in action selection)
anti_stuck_temp = 0.8 # Temperature when stuck
```

### **Common Issues and Solutions**

1. **Agent stuck on walls**: The agent moves along walls instead of getting truly stuck. This is exploration behavior when no targets have been found yet. Once a target is detected, scent gradients guide navigation.

2. **Poor performance**: Performance varies greatly by seed (random placement). Some seeds place agent near targets (good), others far away (poor exploration needed).

3. **Fields not spreading**: Check diffusion parameters. Default 0.25 diffusion with 4 steps creates ~8-10 cell range.

### **Testing Specific Seeds**

Seed 6 is known to work well:
```bash
python cli.py interactive --H 30 --W 30 --seed 6 --nA 2 --nB 2 --max-steps 200
```

This achieves return +2.18 by collecting all 4 targets.

### **Field Visualization**

The HTML viewer shows 6 field panels:
- **GA/GB**: Scent gradients (should spread from targets)
- **P_eff**: Effective potential (combined field for action selection)
- **Vtrail**: Visit trail (builds up in visited areas)
- **Novel**: Novelty field (spikes on observation changes)
- **Ssum**: Schema field (if enabled)

### **Making Changes**

1. **Core algorithms**: `efi/core/fields.py` - action selection, trail updates
2. **Agent logic**: `efi/agents/chemotaxis_agent.py` - field updates, sensing
3. **Runner**: `efi/evaluation/runner.py` - episode loop, field combination
4. **Configs**: `efi/configs/` - default parameters

### **Design Constraints**

- **No global fields**: Don't add fields that require knowledge beyond sensing range
- **Local rules only**: All updates should be based on local information
- **Emergent behavior**: Complex navigation should emerge from simple field interactions
- **Continuous adaptation**: No discrete state machines or mode switches

### **Locality Budget**

Every internal operator is either a single radius-1 stencil pass per tick or
an explicitly iterated relaxation with a declared iteration count k (its
light cone: information propagates at most k cells per call). No global
transforms (scipy EDT, flood fill) inside agent internals; metrics code in
`efi/evaluation/` may use global truth (it measures, it doesn't decide).

| Operator | Where | Radius per call |
|---|---|---|
| `diffuse_masked` | core/diffusion.py | `steps` (1-4 typical) |
| `logodds_predict` | core/belief.py | 1 |
| `value_sweeps` | core/desirability.py | `sweeps` (= kappa, 3 typical; H+W on episode start) |
| `maxplus_distance` | core/localdist.py | `iters` (declared per call site) |
| membrane shells | core/membrane.py | `ceil(max_radius)+1` |
| frontier reachability | core/fields.py | `H+W` (deliberate full-map budget) |
| trail/novelty updates | core/fields.py | 1 |

### **Future Work**

- Schema learning for long-term adaptation
- Multi-agent coordination through pheromone fields  
- Hybrid LLM integration for goal-setting
- Real robot deployment with local sensing only

