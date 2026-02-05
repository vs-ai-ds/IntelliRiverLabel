# IntelliRiverLabel

Places a river name label **inside** the river polygon (Phase A) with safe padding and readable placement. Outputs: `before.png`, `after.png`, `debug.png`, and `placement.json` to `reports/<run_name>/`.

## What it does

- **Phase A (default):** Internal straight placement with padding; label fully inside the polygon with optional rotation.
- Input geometry: `docs/assets/problem/Problem_1_river.wkt` (repo-relative).
- Outputs: `before.png`, `after.png`, `debug.png`, `placement.json` under `reports/<run_name>/`.

## Setup (Windows)

From the repo root:

1. Create the virtual environment: `py -m venv .venv` or `python -m venv .venv`
2. Install dependencies: `pip install -r requirements.txt` (or `.\.venv\Scripts\pip install -r requirements.txt` if not activated)

## Run CLI

Using explicit venv Python (recommended):

```powershell
.\.venv\Scripts\python -m app.core.runner --text "ELBE" --font-size-pt 12 --run-name demo_01
```

## Run Streamlit

Run from the **repo root** (e.g. `D:\HackArena3.0\IntelliRiverLabel`) so the `app` package is found:

```powershell
cd D:\HackArena3.0\IntelliRiverLabel
set PYTHONPATH=%CD%
.\.venv\Scripts\python -m streamlit run app/ui/app.py
```

Or on PowerShell:
```powershell
cd D:\HackArena3.0\IntelliRiverLabel
$env:PYTHONPATH = $PWD
.\.venv\Scripts\python -m streamlit run app/ui/app.py
```

## Outputs

All run artifacts go to **`reports/<run_name>/`** (e.g. `reports/demo_01/`):

- `before.png`, `after.png`, `debug.png`
- `placement.json`
- `run_metadata.json`

## Documentation

- [Project spec](docs/PROJECT_SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Algorithm](docs/ALGORITHM.md)
- [Placement schema](docs/PLACEMENT_SCHEMA.md)
- [Evaluation](docs/EVALUATION.md)
- [UI spec](docs/UI_SPEC.md)

## License

MIT — see [LICENSE](LICENSE).
