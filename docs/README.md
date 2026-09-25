# EFI documentation and demos

EFI explores embodied learning through local fields on a CPU. Start with a
recording, then read the reports for the measured results, controls, and
limits of each experiment.

## Watch a demo

| Recording | What to look for | Open the file |
|---|---|---|
| Continuous contact learning · 180 moves, about 90 seconds | One agent starts with empty evidence, encounters obstacles, and learns through two changes in object response. Jump to move 120 and step forward. | [Long replay](assets/interactive/interaction_long.html) |
| Controlled contact trials · 28 frames | Selected source contacts, then acquired-versus-empty target attempts. These are separate scenes, with omitted source interventions labeled. | [Short replay](assets/interactive/interaction.html) |
| Command consequences · 56 frames | Two commands hold the body still but move the block differently. Six chapter groups compare acquired and command-blind evidence. | [Command replay](assets/interactive/command_contact.html) |
| Returning-context memory · complete lifetime | A bounded model bank learns across five response segments. Recall improves, but the autonomous behavior gate remains open. | [Context replay](assets/interactive/context_memory.html) |

All four use the EFI HTML episode viewer, with recorded fields, action
probabilities, a synchronized probe, and playback controls. Narrated
recordings include a legend, sensing boundaries, chapter buttons, and action
feedback. None of the replays needs Python, a GPU, a server, or an internet
connection. Learning happened during recording; playback does not run it again.

**From a checkout:** open any HTML file in a browser. **On GitHub:** open
the file link, use **Download raw file**, then open the downloaded `.html`.
GitHub displays HTML source rather than running the player. Download linked
players into the same folder if you want their links to each other to work.

To browse the whole research site locally, run this from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Then open [localhost:8000](http://localhost:8000/). [GITHUB_PAGES.md](GITHUB_PAGES.md)
describes how to publish the same site with GitHub Pages.

## Run the agents yourself

After [installation](../README.md#installation), run commands from the repository root:

```bash
# Original foraging controller; open runs/interactive_latest.html
python cli.py interactive

# Longer continuous contact example; open runs/contact-demo/episode.html
python cli.py contact-demo --seed 6 --max-steps 180 --out runs/contact-demo

# Quick command-learning run; open runs/command-smoke/episode.html
python cli.py command-contact --seeds 1 --episodes 2 --seed 31010 --out runs/command-smoke

# Quick returning-context run; open runs/context-smoke/episode.html
python cli.py context-memory --seeds 1 --seed 71020 --out runs/context-smoke

# Quick predictive crossing run; open runs/crossing-smoke/episode.html
python cli.py crossing --seeds 2 --episodes 4 --out runs/crossing-smoke

# Quick motion-transfer run; open runs/transfer-smoke/episode.html
python cli.py transfer --seeds 2 --episodes 2 --acquisition 4 --out runs/transfer-smoke
```

The smoke runs exercise the implementation; they are not held-out estimates.
Each report gives the full evaluation commands and links its archived results.
The learning controllers are opt-in and separate from one another; they have
not been unified into one agent with all earlier capabilities.

## Read the research

Reports are listed roughly in the order the work was done.

| Report | Covers |
|---|---|
| [Mathematical formalization](MATHEMATICAL_FORMALIZATION.md) | Formal definitions for the original chemotaxis-style field controller |
| [Earlier experiment report](EXPERIMENT_REPORT.md) | Historical parameter studies of the chemotaxis controller |
| [Foraging theory](THEORY.md) | The current foraging controller: a linearly-solvable-MDP value recursion computed with local field operations |
| [Fixed-point tracking](TRACKING.md) | Why a few value sweeps per tick keep up with slowly changing beliefs |
| [Non-stationary worlds](EXPERIMENTS_NONSTAT.md) | Pre-registered hypotheses and results for drifting, regrowing, and reward-swapping worlds |
| [Predictive crossing](PREDICTIVE_CONTROL.md) | Learning to anticipate moving hazards and adapting after their motion changes |
| [Motion transfer](PREDICTIVE_TRANSFER.md) | Reusing learned motion across object roles and room geometry |
| [Contact learning](INTERACTION_LEARNING.md) | Learning what contact does to an object, with held-out controls, CPU/memory costs, and a viewer walkthrough |
| [Command consequences](COMMAND_CONSEQUENCES.md) | Learning what each command changes when commands produce identical body motion |
| [Returning-context memory](CONTEXT_MEMORY.md) | A bounded memory of response models: a recall gain, a failed behavioral gate, and capacity limits |

The last two experiments each followed a protocol written before any held-out
runs: [command consequences](EFI01_PROTOCOL.md) and
[returning-context memory](EFI02_PROTOCOL.md). These files are kept exactly
as they were locked, because the validation scripts check their hashes, so
they still use the project's internal experiment labels.

Each report links its raw trials, summaries, and validation records under
`assets/data/`. Recorded demonstrations illustrate behavior; statistical
claims come from the complete evaluations, including failures.

## Maintain the site

`index.html` is the research landing page. `assets/interactive/` contains
standalone players; `assets/images/` holds figures and GIFs; `assets/data/`
holds archived measurements. Styles and chart code live in `assets/css/`
and `assets/js/`. [GIF_EXPORT.md](GIF_EXPORT.md) explains how to export
episodes as animated GIFs.

When updating an experiment, use the reproduction commands in its report
and keep its methods, data, figures, and validation record together.
The landing page labels its older chemotaxis charts as historical results.
