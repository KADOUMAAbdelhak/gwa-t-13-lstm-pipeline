# DynAPM Task 5A — Empty-window root-cause diagnostic

## Diagnosis findings

The hypothesis is **partially supported but not fully confirmed**.

- **Confirmed part:** Missing-sample gaps clearly exist and can directly produce empty 24h forward windows (worst case VM 400 shows a single 255.08h gap covering all 19 skipped evaluation points).
- **Not confirmed as sole corpus-wide explanation:** correlation between per-VM skip count and per-VM count of gaps >24h is **0.248** (low, well below 0.7).

So: data gaps are definitely a cause, but not the only factor driving skip counts across all VMs.

---

## 1) Top-10 affected VMs: sampling-interval gap stats

| trace | vm_id | mode delta (s) | count(delta!=300s) | longest gap (h) | longest gap timestamp | distinct gap-points (delta>600s) | longest gap in 24h units | skipped windows |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Materna-Trace-1 | 400 | 300.0 | 6 | 255.08 | 2015-11-30 15:00:00 | 6 | 10.6285 | 19 |
| Materna-Trace-3 | 364 | 300.0 | 4 | 94.58 | 2016-01-28 22:30:00 | 4 | 3.9410 | 6 |
| Materna-Trace-3 | 281 | 300.0 | 23 | 61.08 | 2016-01-24 13:00:00 | 23 | 2.5451 | 3 |
| Materna-Trace-1 | 197 | 300.0 | 1 | 48.08 | 2015-11-16 00:00:00 | 1 | 2.0035 | 2 |
| Materna-Trace-2 | 209 | 300.0 | 2 | 48.08 | 2015-12-15 00:00:00 | 2 | 2.0035 | 2 |
| Materna-Trace-1 | 278 | 300.0 | 2 | 35.25 | 2015-11-27 11:10:00 | 2 | 1.4688 | 1 |
| Materna-Trace-2 | 104 | 300.0 | 1 | 24.08 | 2015-12-22 00:00:00 | 1 | 1.0035 | 1 |
| Materna-Trace-2 | 132 | 300.0 | 1 | 24.08 | 2015-12-19 00:00:00 | 1 | 1.0035 | 1 |
| Materna-Trace-2 | 167 | 300.0 | 1 | 24.08 | 2015-12-20 00:00:00 | 1 | 1.0035 | 1 |
| Materna-Trace-2 | 174 | 300.0 | 5 | 24.08 | 2015-12-31 00:00:00 | 4 | 1.0035 | 1 |

Observation: all ten keep 5-minute mode, but each has at least one large interruption; VM 400 has an extreme multi-day break.

---

## 2) Zoomed inspection: Trace-1 / VM 400

All empty-window evaluation timestamps for this VM (19 total):

- 2015-11-20 09:40:00
- 2015-11-20 21:40:00
- 2015-11-21 09:40:00
- 2015-11-21 21:40:00
- 2015-11-22 09:40:00
- 2015-11-22 21:40:00
- 2015-11-23 09:40:00
- 2015-11-23 21:40:00
- 2015-11-24 09:40:00
- 2015-11-24 21:40:00
- 2015-11-25 09:40:00
- 2015-11-25 21:40:00
- 2015-11-26 09:40:00
- 2015-11-26 21:40:00
- 2015-11-27 09:40:00
- 2015-11-27 21:40:00
- 2015-11-28 09:40:00
- 2015-11-28 21:40:00
- 2015-11-29 09:40:00

For **every** timestamp above:

- closest row before `t`: **2015-11-19 23:55:00**
- closest row after `t`: **2015-11-30 15:00:00**
- gap spanning `t`: **255.0833 hours**

This is a direct, deterministic explanation for VM 400’s skips.

---

## 3) Corpus-wide gap characterization (1,576 kept VMs)

- VMs with at least one gap > 30 min: **342**
- VMs with at least one gap > 1 hour: **265**
- VMs with at least one gap > 24 hours: **243**

Largest-gap-per-VM (hours) distribution:

- min: **0.0833**
- p50: **0.0833**
- p95: **24.0833**
- p99: **24.0833**
- max: **255.0833**

Figure:

- `dynapm/reports/task_05a/largest_gap_histogram.png`

---

## 4) Cross-validation with skip count

Computed correlations:

- corr( `n_skipped_windows`, `count(gaps>24h)` ) = **0.2477**
- corr( `n_skipped_windows`, `has_any_gap>24h` ) = **0.2852**

Interpretation under requested criterion:

- Since correlation is not high (>0.7), "gaps alone explain skips" is **not** confirmed corpus-wide.

---

## 5) Evaluation-clock sanity check (Trace-1 / 501)

Requested checks yielded:

- eval interval exactly 12h: **True**
- all eval timestamps <= `streaming_end - 24h`: **True**
- all eval timestamps on a data row or within 5 min: **False**
  - max nearest-row distance: **34,800s (9.67h)**

Additional finding for this VM:

- `Trace-1/501.csv` has one **24.0833h** gap (`count(delta!=300)=1`), so this "healthy" example is not gap-free; two eval points land inside/near this gap.

This indicates the clock logic itself is consistent, but gapful data can violate the row-proximity expectation.

---

## Recommendation

Use a **data-aware evaluation clock** (or pre-filter eval timestamps) that only emits timestamps whose `[t, t+24h)` window intersects sufficient actual observations, instead of assuming regular 5-minute continuity.

---

## Files created or modified

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05a/largest_gap_histogram.png`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05a/diagnosis_summary.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/data/empty_window_investigation.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05a_empty_window_diagnosis.md`

---

## Commands (output truncated)

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python - << 'PY' ...
{
  "summary_json": "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05a/diagnosis_summary.json",
  "histogram": "/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_05a/largest_gap_histogram.png",
  "skip_total": 106,
  "corr": 0.24768116554327427
}
```

```text
$ cd /Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline && dynapm/.venv/bin/python - << 'PY' ... (Trace-1/501 sanity deep-check)
max_gap_h 24.083333333333332 count_gt300 1
n_eval 40 start 2015-11-13 09:35:00 end 2015-12-02 21:35:00 stream_start 2015-11-13 09:35:00
worst3
34800.0 2015-11-30 09:35:00 2015-11-29 23:55:00 2015-12-01 00:00:00
8700.0 2015-11-30 21:35:00 2015-11-29 23:55:00 2015-12-01 00:00:00
0.0 2015-12-02 21:35:00 2015-12-02 21:30:00 2015-12-02 21:35:00
```

---

## Assumptions

- Empty window definition is exactly `derive_ground_truth_at_t` raising `ValueError('empty window ...')`.
- Gap point definition follows your spec: timestamp delta > 600s.
- For VM 400 zoom, "gap encompassing t" is measured between nearest row before and nearest row after `t`.
- Correlation criterion uses Pearson correlation, as standard for scalar-vs-scalar linear association.

---

## Surprises

- The nominally "healthy" `Trace-1/501` contains a 24h+ gap, invalidating the row-proximity expectation for all its eval timestamps.
- The skip issue is highly concentrated (VM 400 contributes 19/106) but still present across many VMs (79), with low correlation against just count(gaps>24h), implying timing/alignment effects also matter.
