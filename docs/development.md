# Development

For whoever changes PhysLoc itself. Back to the [README](../README.md); design reasoning is in
[PLAN.md](PLAN.md), the sample layout in [schema.md](schema.md).

## Two environments

| | runs | contains |
|---|---|---|
| **container** — pinned Kubric image | scene sampling, simulation, rendering | Kubric 2022.4.1, Blender 2.93.4, PyBullet, Python 3.9 |
| **host** — `conda activate physloc` | annotation, residuals, masks, validation, visualisation | numpy, scipy, opencv, jsonschema, Python 3.11 |

They meet at the trajectory seam, `traj.npz`; never install Kubric, Blender or PyBullet on the
host. `docker/kubric.sh <script.py>` runs a script from this repo inside the container.
`bash scripts/fetch_refs.sh` checks out a read-only copy of the Kubric source for reference.

## Keeping the tables honest

The taxonomy, ladder, condition, difficulty and material tables in the [README](../README.md),
and the tier and cost tables in [generating.md](generating.md), are **generated** from
`physloc/taxonomy.py`, `physloc/scenarios/base.py` and `physloc/cli.py`. `reference.BLOCK_FILES`
says which document holds which block. After changing any of the sources:

```bash
python -m physloc.reference            # is every document current? exits 1 if any is stale
python -m physloc.reference --write    # regenerate the tables in place
```

`tests/test_reference.py` fails when any is stale. The HuggingFace card is generated from the same
functions (`reference.hf_card`), minus the cost tables, which price configs by running the CLI.

## Tests

```bash
python -m pytest tests                 # the full suite: about 40 minutes, pour cells are slowest
python -m pytest tests/test_reference.py tests/test_cpu_slots.py   # a quick subset
```

## Repository layout

```
physloc/scenarios/    scenario builders, the complexity ladder and conditions (base.py)
physloc/injectors/    the violation families, one file per domain
physloc/render/       the container worker, and probes for render cost
physloc/sim/          trajectories and the simulation seam
physloc/residuals/    the physical residuals severity is measured from
physloc/annotate/     residuals -> masks, severity, clocks, and schema-v4 samples
physloc/loader.py     reads a release and derives every annotation (numpy only)
physloc/release/      export, splits, dataset card
physloc/viz/          the overlay renderer, browser viewer, grids, sheets; every mp4
test_dataset_loader.py  load a dataset, print its structure, look at it
physloc/cli.py        the `physloc` command line
configs/              common.yaml and one file per run
scripts/              run.sh, run_fast.sh, probes and refresh tools
docs/schema.md        the sample layout, fields, axes, units, and loader contract
docs/performance.md   how cost and performance were measured
docs/PLAN.md          the design document
docs/roadmap.md       what is next
docs/generating.md    configs, running, costs
docs/publishing.md    validate, export, upload
docs/development.md   environments, tests, layout
```

