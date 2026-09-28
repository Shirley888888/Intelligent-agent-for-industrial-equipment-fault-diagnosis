"""modify:audit-output-isolation. Reproductions cannot overwrite the frozen experiment."""
from pathlib import Path
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent

def reproduction_dir(kind, requested=None):
    out = Path(requested).expanduser() if requested else ROOT / "results" / "reruns" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + kind)
    if not out.is_absolute(): out = ROOT / out
    out = out.resolve()
    # Block both formal paths and a parent directory which could replace them.
    protected = [ROOT / "results/final", ROOT / "paper", ROOT / "archive"]
    if any(out == p.resolve() or p.resolve() in out.parents or out in p.resolve().parents for p in protected):
        raise ValueError("Reproduction output must not overwrite final, paper or archive paths.")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty reproduction directory.")
    out.mkdir(parents=True, exist_ok=True)
    return out
