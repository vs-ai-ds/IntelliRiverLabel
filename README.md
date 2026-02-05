# IntelliRiverLabel

Places a river name label **inside a river polygon** (WKT) with safe padding and readable placement. Exports `before.png`, `after.png`, `debug.png`, and `placement.json` to `reports/<run_name>/`.

## Setup (Windows)

From the repo root (e.g. `D:\HackArena3.0\IntelliRiverLabel`):

**1. Create the virtual environment** (use either):

```powershell
py -m venv .venv
```

or

```powershell
python -m venv .venv
```

**2. Activate the environment** (optional but convenient):

- **PowerShell:**  
  `.venv\Scripts\Activate.ps1`

- **Command Prompt:**  
  `.venv\Scripts\activate.bat`

**3. Install dependencies** (use one of these):

- If activated:  
  `pip install -r requirements.txt`

- If not activated:  
  `.\.venv\Scripts\pip install -r requirements.txt`

(Use `-r` so pip reads the list from `requirements.txt`.)

## Run

- **With activation:**  
  `python -m app.core.runner`

- **Without activation:**  
  `.\.venv\Scripts\python -m app.core.runner`

Example with options:

```powershell
.\.venv\Scripts\python -m app.core.runner --text "ELBE" --font-size-pt 12 --run-name demo_01
```

Outputs: `reports/demo_01/before.png`, `after.png`, `debug.png`, `placement.json`.

## Inputs

Challenge assets (repo-relative):

- `docs/assets/problem/Problem_1_river.wkt`
- `docs/assets/problem/problem_1.pdf`
- `docs/assets/problem/problem_1_blurb.jpeg`

## Documentation

- [Project spec](docs/PROJECT_SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Algorithm](docs/ALGORITHM.md)
- [Placement schema](docs/PLACEMENT_SCHEMA.md)
- [Evaluation](docs/EVALUATION.md)
- [UI spec](docs/UI_SPEC.md)

## License

MIT — see [LICENSE](LICENSE).
