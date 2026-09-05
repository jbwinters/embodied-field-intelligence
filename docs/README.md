# EFI documentation and demos

EFI explores embodied learning through local fields on a CPU. Start with a
recording, then use the reports to distinguish measured results from the
next architectural proposals.

## Watch a demo

| Recording | What to look for | Open the file |
|---|---|---|
| Continuous contact learning · 180 moves, about 90 seconds | One agent starts with empty evidence, encounters obstacles, and learns through two changes in object response. Jump to move 120 and step forward. | [Long replay](assets/interactive/interaction_long.html) |
| Controlled contact trials · 28 frames | Selected source contacts, then acquired-versus-empty target attempts. These are separate scenes, with omitted source interventions labeled. | [Short replay](assets/interactive/interaction.html) |

These are the original EFI HTML viewer, with recorded fields, action
probabilities, a synchronized probe, and playback controls. The longer
recording adds a legend, sensing boundaries, chapter buttons, and action
feedback. Neither replay needs Python, a GPU, a server, or an internet
connection. Learning happened during recording; playback does not run it again.

**From a checkout:** open either HTML file in a browser. **On GitHub:** open
the file link, use **Download raw file**, then open the downloaded `.html`.
GitHub displays HTML source rather than running the player. Download both
files into the same folder if you want their links to each other to work.

To browse the entire research site locally, run this from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open [the local site](http://localhost:8000/) or
[the longer player](http://localhost:8000/assets/interactive/interaction_long.html).
The [public site](https://jbwinters.github.io/embodied-field-intelligence/)
requires a GitHub Pages deployment; an unmerged branch is not published there.
See [site deployment](GITHUB_PAGES.md).

## Run the agents yourself

After [installation](../README.md#installation), run commands from the repository root:

```bash
# Longer continuous contact example; open runs/contact-demo/episode.html
python cli.py contact-demo --seed 6 --max-steps 180 --out runs/contact-demo

# Original foraging controller; open runs/interactive_latest.html
python cli.py interactive

# Quick predictive crossing run; open runs/crossing-smoke/episode.html
python cli.py crossing --seeds 2 --episodes 4 --out runs/crossing-smoke

# Quick motion-transfer run; open runs/transfer-smoke/episode.html
python cli.py transfer --seeds 2 --episodes 2 --acquisition 4 --out runs/transfer-smoke
```

The smoke runs exercise the implementation. The reports below provide the
full evaluation commands and archived results. The new controllers remain
opt-in; they have not been unified into one agent with all earlier capabilities.

## Read the research

| Start with | Covers |
|---|---|
| [Contact learning](INTERACTION_LEARNING.md) | Current implementation, a viewer walkthrough, held-out controls, CPU/memory costs, and limitations |
| [Predictive crossing](PREDICTIVE_CONTROL.md) | Learning to anticipate moving hazards and adapt after their motion changes |
| [Motion transfer](PREDICTIVE_TRANSFER.md) | Reusing acquired motion across object roles and room geometry |
| [Online intelligence design](ONLINE_INTELLIGENCE_DESIGN.md) | Architecture and staged gates; contact is implemented, recurring-context retention and learned-skill composition remain proposals |
| [Independent design review](ONLINE_INTELLIGENCE_REVIEW.md) | Critiques and the design's responses |
| [Foraging theory](THEORY.md) | The current foraging value recursion and its local implementation |
| [Earlier experiment report](EXPERIMENT_REPORT.md) | Historical experiments; these are separate from the newer pilots |

Each capability report links its raw trials, summaries, and validation
records under `assets/data/`. Recorded demonstrations illustrate behavior;
statistical claims come from the complete evaluations, including failures.

## Maintain the site

`index.html` is the research landing page. `assets/interactive/` contains
standalone players; `assets/images/` holds figures and GIFs; `assets/data/`
holds archived measurements. Styles and chart code live in `assets/css/`
and `assets/js/`.

Use the reproduction commands in the relevant report when updating an
experiment. Keep its methods, data, figures, and validation record together.
The landing page labels its older chemotaxis charts as historical results.
