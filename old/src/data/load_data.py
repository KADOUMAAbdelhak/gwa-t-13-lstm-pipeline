from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

TRACE_INFO_FILENAME = "00-Trace-Info.txt"


def list_trace_dirs(root: str, traces: Optional[Iterable[str]] = None) -> List[Path]:
    """
    Return the list of trace directories that should be processed.
    If `traces` is None, all directories that match the Materna naming scheme are returned.
    """
    root_path = Path(root)
    if traces:
        return [root_path / t for t in traces]
    return sorted([p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("Materna-Trace-")])


def list_vm_files(trace_dir: Path) -> List[Path]:
    """
    Return all VM CSV files for a given trace directory.
    The Materna traces contain a `00-Trace-Info.txt` metadata file that should be ignored.
    """
    return sorted([p for p in trace_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def robust_read_csv(path: Path) -> pd.DataFrame:
    """
    Read a Materna CSV file which is semicolon separated and may contain comment lines.
    """
    for sep in (";", ","):
        try:
            return pd.read_csv(path, sep=sep, engine="python", header=None, comment="#", encoding="utf-8")
        except Exception:
            continue
    # fallback: manual parsing
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = [p.strip() for p in ln.strip().split(";")]
            if parts and any(parts):
                rows.append(parts)
    return pd.DataFrame(rows)


def load_vm_to_df(path: Path) -> pd.DataFrame:
    """Returns the raw dataframe for a single VM without modifying columns."""
    return robust_read_csv(Path(path))


def load_trace_info(trace_dir: Path) -> Optional[str]:
    """Read the optional 00-Trace-Info.txt metadata file if present."""
    info_path = trace_dir / TRACE_INFO_FILENAME
    if not info_path.exists():
        return None
    return info_path.read_text(encoding="utf-8", errors="ignore")


def load_data(root: str, traces: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Convenience helper primarily for notebooks/experimentation.
    Loads all VM CSVs for the requested traces into memory, grouped by trace -> VM filename.
    """
    dataset: Dict[str, Dict[str, pd.DataFrame]] = {}
    for trace_dir in list_trace_dirs(root, traces):
        if not trace_dir.exists():
            continue
        vm_data = {}
        for vm_file in list_vm_files(trace_dir):
            vm_data[vm_file.name] = load_vm_to_df(vm_file)
        dataset[trace_dir.name] = vm_data
    return dataset