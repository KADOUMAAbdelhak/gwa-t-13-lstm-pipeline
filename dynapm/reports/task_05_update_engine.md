# DynAPM Task 5 — Update engine, decision head, churn controller, baselines, 100-VM run

## Status

Execution was **intentionally stopped at Subtask 2** per instruction:

> "If symptomatic, stop and report."

The empty-window investigation result was **symptomatic** (not benign tail-only skips), so subtasks 3–12 were **not executed**.

---

## Schema update: diurnal_pattern removed (Subtask 1 completed)

### Changes applied

- `dynapm/core/schema.py`
  - Removed `diurnal_pattern` from `DynamicFields`.
  - Removed `DIURNAL_PATTERNS` constant.
  - Changed `DynamicFields` to `@dataclass(frozen=False)` (required later by UpdateEngine design).
  - Updated `ProfileInstance.to_json()` to no longer emit `diurnal_pattern`.
- `dynapm/core/ground_truth.py`
  - Removed `diurnal_pattern` from `derive_ground_truth_at_t()` output.
  - Removed corresponding import and doc lines.
- `dynapm/core/pattern.py`
  - Kept `derive_diurnal_pattern()` utility.
  - Added docstring note: not part of active dynamic-field set after Task 5.
- `dynapm/tests/test_profile_split.py`
  - Updated `test_ground_truth_returns_all_fields` to expect 4 keys.
  - Kept synthetic diurnal utility test.
- `dynapm/scripts/task04_ground_truth_stats.py`
  - Removed dependency on `diurnal_pattern` field for aggregate stats JSON.

### Validation

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py tests/test_predictor.py tests/test_drift.py tests/test_evaluation_clock.py -q
.................                                                        [100%]
17 passed in 7.90s
```

### Ground-truth regeneration after schema change

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py
{
  "n_total": 1576,
  "n_ok": 1576,
  "n_failed": 0,
  "wall_seconds": 30.877670958987437,
  "mean_eval_points": 41.83565989847716,
  "total_skipped_empty_windows": 106
}
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth_stats.py
{
  "stats_json": "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_stats.json",
  "n_points": 65933
}
```

**Consistency check:** `n_points` remained **65933** (unchanged).

---

## Empty-window investigation (Subtask 2 completed)

Diagnostic output (`dynapm/data/empty_window_investigation.json`):

- total candidate eval timestamps across corpus: **66039**
- skipped windows: **106**
- skipped fraction: **0.001605** (~0.1605%)
- VMs with at least one skipped window: **79**
- VMs with non-tail skipped windows: **79**
- non-tail skips total: **106**

### Required answers

1. **Fraction skipped** across all candidate eval timestamps:
   - **106 / 66039 = 0.001605** (~0.1605%)

2. **Concentration vs spread**:
   - Not uniform.
   - Concentrated in a subset of VMs (79/1576), with a strong hotspot (`Materna-Trace-1 / VM 400` has 19 skips).

3. **Tail-only benign vs elsewhere symptomatic**:
   - **Symptomatic**: skips are **not** confined to streaming tail.
   - Non-tail skips = **106/106**.

4. **Top 10 VMs by skipped windows + streaming durations**:

| trace | vm_id | skipped | non_tail | streaming_days |
|---|---:|---:|---:|---:|
| Materna-Trace-1 | 400 | 19 | 19 | 23.5938 |
| Materna-Trace-3 | 364 | 6 | 6 | 15.6319 |
| Materna-Trace-3 | 281 | 3 | 3 | 26.0764 |
| Materna-Trace-1 | 197 | 2 | 2 | 20.8993 |
| Materna-Trace-2 | 209 | 2 | 2 | 22.5972 |
| Materna-Trace-1 | 278 | 1 | 1 | 14.0660 |
| Materna-Trace-2 | 104 | 1 | 1 | 21.9965 |
| Materna-Trace-2 | 132 | 1 | 1 | 21.9965 |
| Materna-Trace-2 | 167 | 1 | 1 | 21.9965 |
| Materna-Trace-2 | 174 | 1 | 1 | 21.9965 |

### Stop condition triggered

Because skips are symptomatic (not benign tail-only), execution stops here by task instruction. Subtasks 3–12 were not started.

---

## Files created or modified (absolute paths)

### Modified

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/schema.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/ground_truth.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/core/pattern.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/tests/test_profile_split.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task04_ground_truth_stats.py`

### Generated during reruns/diagnostic

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_run_meta.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_failures.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_stats.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/empty_window_investigation.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05_update_engine.md`

---

## Assumptions

- `candidate_eval_total` is computed as the total count of timestamps from `compute_evaluation_timestamps(streaming_index, 12h)` across all kept VMs.
- A skipped window is any eval timestamp where `derive_ground_truth_at_t(...)` raises `ValueError("empty window ...")`.
- A skip is marked **non-tail** if it is not the final eval timestamp in that VM’s candidate list.

---

## Deviations

- Task was **not** completed through subtask 12 because the explicit stop condition in subtask 2 was triggered (symptomatic skips).

---

## Surprises

- Even after schema-only changes and successful full rerun (`n_ok=1576`), skipped windows persisted exactly at **106**, and all were non-tail.
- Skip concentration is highly uneven, with one VM contributing a large share.

---

## Commands (output truncated to <=20 lines each)

```text
$ python3 - << 'PY' ... (apply schema/ground_truth/pattern/test/gt-stats edits)
subtask1 schema/gt updates applied
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm && .venv/bin/pytest tests/test_profile_split.py tests/test_predictor.py tests/test_drift.py tests/test_evaluation_clock.py -q
.................                                                        [100%]
17 passed in 7.90s
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth.py
{
  "n_total": 1576,
  "n_ok": 1576,
  "n_failed": 0,
  "wall_seconds": 30.877670958987437,
  "mean_eval_points": 41.83565989847716,
  "total_skipped_empty_windows": 106
}
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python dynapm/scripts/task04_ground_truth_stats.py
{
  "stats_json": "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/ground_truth_stats.json",
  "n_points": 65933
}
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python - << 'PY' ... (empty-window investigation)
{
  "candidate_eval_total": 66039,
  "skipped_total": 106,
  "skipped_fraction": 0.0016051121307106408,
  "n_vms_with_skips": 79,
  "n_vms_with_non_tail_skips": 79,
  "non_tail_skips_total": 106,
  "top10_skipped_vms": [...]
}
```

---

## Next-step suggestion

Before continuing with Task 5 subtasks 3–12, investigate why `derive_ground_truth_at_t` still returns empty windows for non-tail evaluation timestamps on 79 VMs (likely timestamp-index gaps/misalignment between evaluation clock and source DataFrame windows), fix that root cause deterministically, and only then proceed to implement/update-engine and 100-VM method comparisons to avoid training/evaluation labels being silently missing in non-terminal regions.
