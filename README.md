# Traction / Resistance Curves Tool

Desktop utility in Python for plotting and inspecting:

- Straight resistance
- Slope resistance
- Total resistance
- Raw traction effort
- Effective traction effort (limited by adhesion when enabled)

The tool uses `Calculations.dll` through `pythonnet`.

## Repository contents

This repository already includes:

- `traction-resistance.py` — main application
- `rail_resistance_config.json` — example/default configuration
- `Calculations.dll` — required .NET assembly
- `.gitignore`

## Requirements

- Python 3.10+ recommended
- `pip`
- .NET SDK installed on the machine
- On Linux/WSL: a working `dotnet` command available in PATH

## Quick start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate a virtual environment

#### Linux/WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install numpy matplotlib pythonnet
```

### 4. Verify .NET is installed

```bash
dotnet --info
```

If this command fails, install the .NET SDK first.

### 5. Run the application

```bash
python traction-resistance.py
```