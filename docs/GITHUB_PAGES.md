# Publish and preview the research site

The site uses the existing `docs/index.html`, CSS, charts, and standalone
HTML viewers. No separate app build is required. See the
[documentation index](README.md) for recordings and research reports.

## Preview a checkout

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open [localhost:8000](http://localhost:8000/) and follow the **Demos** link.
The individual HTML players also open directly from disk and work offline.
The landing page's interactive charts use Chart.js from a CDN; archived
figures remain visible when that library or the chart data cannot load.

## Publish through GitHub Pages

After the reviewed changes are merged into `main`, configure the repository:

1. Open **Settings → Pages**.
2. Select **Deploy from a branch**.
3. Select the **main** branch and **/docs** folder, then save.
4. Wait for the Pages deployment to succeed in **Actions**.

The intended public address is
[the EFI research site](https://jbwinters.github.io/embodied-field-intelligence/).
Once deployed, the players are available at
[the continuous example](https://jbwinters.github.io/embodied-field-intelligence/assets/interactive/interaction_long.html)
and [the controlled trials](https://jbwinters.github.io/embodied-field-intelligence/assets/interactive/interaction.html).
A feature branch does not update this deployment. A 404 before deployment
does not prevent opening the same files locally.

## Check an update

- Follow the landing page's demo links and test playback, seeking, chapter
  jumps, and a narrow browser window.
- Confirm that each chart uses its archived data and each research link
  points to `main`, so deleting a merged feature branch does not break it.
- After deployment, open both public player URLs and check the Pages job.

The site contains historical chemotaxis charts as well as newer capability
reports. Keep those experiment labels intact when replacing artifacts; do
not substitute illustrative fallback numbers for missing measurements.
