# DynAPM Task 0 — Materna corpus characterization

## Meta (protocol)

- **What was done**: Created `dynapm/`, `dynapm/reports/`, `dynapm/reports/task_00/`, and `dynapm/scripts/`; added `task00_characterize.py` (pandas/matplotlib implementation per spec); verified dependencies missing on this machine; ran an equivalent stdlib-only corpus scan (~58 s) writing `characterization_fallback.json`; assembled this discovery report with directory inventory, Trace-Info excerpts (first 500 lines × 3), three exemplar CSV profiles, corpus metrics, target-column rationale, 100-VM distributional stats, missing-data rates, environment check, and disk/CPU/RAM facts.
- **Written outputs**: `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task00_characterize.py`, `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_00/characterization_fallback.json`, `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_00_discovery.md`.
- **Data sources read** (read-only): Materna traces under `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-{1,2,3}/` (VM `*.csv`, `00-Trace-Info.txt`, plus ancillary files listed in §1).

## Findings

### 1. DIRECTORY INVENTORY

| Trace folder | Absolute path | Files | Total size (MB) | `00-Trace-Info.txt` |
|---|---|---:|---:|---|
| Materna-Trace-1 | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-1` | 521 | 367.38 | yes |
| Materna-Trace-2 | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-2` | 528 | 398.64 | yes |
| Materna-Trace-3 | `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-3` | 548 | 471.17 | yes |

**Extension counts** (from `characterization_fallback.json`):

- **Trace 1**: `.csv` 518, `.txt` 1, `(no ext)` 1, `.bak` 1.
- **Trace 2**: `.csv` 527, `.txt` 1.
- **Trace 3**: `.csv` 547, `.txt` 1.

VM CSV counts only: Trace 1 = 518, Trace 2 = 527, Trace 3 = 547.

### 2. 00-TRACE-INFO.TXT — CRITICAL

Each trace folder contains `00-Trace-Info.txt`. Full files are longer than 500 lines; below are the **first 500 lines** of each (per spec).

#### Materna-Trace-1 (`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-1/00-Trace-Info.txt`)

*(Total lines in file: 10949; excerpt: first 500 lines.)*

```text

TraceAnalysis:


Results for each Workload Trace:

Trace File: 1
CPU values:

Provisioned CPU Cores: 2

- minCPU: 97 MHz
- maxCPU: 5024 MHz
- avgCPU: 379 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 57 MB
- maxRAM: 2457 MB
- avgRAM: 764 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 2
CPU values:

Provisioned CPU Cores: 2

- minCPU: 92 MHz
- maxCPU: 5045 MHz
- avgCPU: 334 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 49 MB
- maxRAM: 2779 MB
- avgRAM: 372 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 3
CPU values:

Provisioned CPU Cores: 2

- minCPU: 127 MHz
- maxCPU: 924 MHz
- avgCPU: 186 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 294 MB
- maxRAM: 3926 MB
- avgRAM: 795 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 4
CPU values:

Provisioned CPU Cores: 2

- minCPU: 69 MHz
- maxCPU: 1862 MHz
- avgCPU: 85 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 632 MB
- maxRAM: 4390 MB
- avgRAM: 1034 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 5
CPU values:

Provisioned CPU Cores: 2

- minCPU: 60 MHz
- maxCPU: 1725 MHz
- avgCPU: 72 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 622 MB
- maxRAM: 4947 MB
- avgRAM: 1007 MB

# of TraceLog entries: 8064

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 6
CPU values:

Provisioned CPU Cores: 1

- minCPU: 23 MHz
- maxCPU: 303 MHz
- avgCPU: 29 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 37 MB
- maxRAM: 1343 MB
- avgRAM: 101 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 7
CPU values:

Provisioned CPU Cores: 1

- minCPU: 21 MHz
- maxCPU: 240 MHz
- avgCPU: 27 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 31 MB
- maxRAM: 1317 MB
- avgRAM: 95 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 8
CPU values:

Provisioned CPU Cores: 4

- minCPU: 34 MHz
- maxCPU: 3396 MHz
- avgCPU: 429 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 114 MB
- maxRAM: 7956 MB
- avgRAM: 584 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 9
CPU values:

Provisioned CPU Cores: 2

- minCPU: 16 MHz
- maxCPU: 2351 MHz
- avgCPU: 70 MHz

RAM values:

Provisioned RAM: 6 GB

- minRAM: 8 MB
- maxRAM: 6070 MB
- avgRAM: 264 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 10
CPU values:

Provisioned CPU Cores: 4

- minCPU: 35 MHz
- maxCPU: 4462 MHz
- avgCPU: 206 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 223 MB
- maxRAM: 7295 MB
- avgRAM: 819 MB

# of TraceLog entries: 8064

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 11
CPU values:

Provisioned CPU Cores: 8

- minCPU: 25 MHz
- maxCPU: 11536 MHz
- avgCPU: 1351 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 86 MB
- maxRAM: 7874 MB
- avgRAM: 2688 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 12
CPU values:

Provisioned CPU Cores: 8

- minCPU: 23 MHz
- maxCPU: 11306 MHz
- avgCPU: 893 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 97 MB
- maxRAM: 7683 MB
- avgRAM: 2132 MB

# of TraceLog entries: 8064

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 13
CPU values:

Provisioned CPU Cores: 8

- minCPU: 17 MHz
- maxCPU: 3933 MHz
- avgCPU: 350 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 16 MB
- maxRAM: 6553 MB
- avgRAM: 1224 MB

# of TraceLog entries: 8064

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 14
CPU values:

Provisioned CPU Cores: 2

- minCPU: 15 MHz
- maxCPU: 1844 MHz
- avgCPU: 85 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 20 MB
- maxRAM: 1835 MB
- avgRAM: 349 MB

# of TraceLog entries: 8064

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 15
CPU values:

Provisioned CPU Cores: 2

- minCPU: 141 MHz
- maxCPU: 3417 MHz
- avgCPU: 644 MHz

RAM values:

Provisioned RAM: 16 GB

- minRAM: 1658 MB
- maxRAM: 13378 MB
- avgRAM: 5570 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 16
CPU values:

Provisioned CPU Cores: 1

- minCPU: 12 MHz
- maxCPU: 2324 MHz
- avgCPU: 241 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 34 MB
- maxRAM: 1786 MB
- avgRAM: 428 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 17
CPU values:

Provisioned CPU Cores: 1

- minCPU: 13 MHz
- maxCPU: 815 MHz
- avgCPU: 29 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 20 MB
- maxRAM: 1585 MB
- avgRAM: 84 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 18
CPU values:

Provisioned CPU Cores: 1

- minCPU: 13 MHz
- maxCPU: 910 MHz
- avgCPU: 30 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 2 MB
- maxRAM: 3402 MB
- avgRAM: 96 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 19
CPU values:

Provisioned CPU Cores: 1

- minCPU: 32 MHz
- maxCPU: 57 MHz
- avgCPU: 41 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 1 MB
- maxRAM: 184 MB
- avgRAM: 39 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 20
CPU values:

Provisioned CPU Cores: 4

- minCPU: 50 MHz
- maxCPU: 157 MHz
- avgCPU: 64 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 92 MB
- maxRAM: 704 MB
- avgRAM: 307 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 21
CPU values:

Provisioned CPU Cores: 2

- minCPU: 28 MHz
- maxCPU: 2997 MHz
- avgCPU: 212 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 16 MB
- maxRAM: 7945 MB
- avgRAM: 901 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 22
CPU values:

Provisioned CPU Cores: 1

- minCPU: 11 MHz
- maxCPU: 2149 MHz
- avgCPU: 128 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 65 MB
- maxRAM: 1337 MB
- avgRAM: 430 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 23
CPU values:

Provisioned CPU Cores: 1

- minCPU: 7 MHz
- maxCPU: 21 MHz
- avgCPU: 7 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 2 MB
- maxRAM: 180 MB
- avgRAM: 13 MB

# of TraceLog entries: 8352

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 24
CPU values:

Provisioned CPU Cores: 4

- minCPU: 150 MHz
- maxCPU: 6151 MHz
- avgCPU: 380 MHz

RAM values:

```

#### Materna-Trace-2 (`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-2/00-Trace-Info.txt`)

*(Total lines in file: 11096; excerpt: first 500 lines.)*

```text

TraceAnalysis:


Results for each Workload Trace:

Trace File: 1
CPU values:

Provisioned CPU Cores: 2

- minCPU: 90 MHz
- maxCPU: 939 MHz
- avgCPU: 119 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 21 MB
- maxRAM: 2512 MB
- avgRAM: 297 MB

# of TraceLog entries: 8918

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 2
CPU values:

Provisioned CPU Cores: 2

- minCPU: 85 MHz
- maxCPU: 2947 MHz
- avgCPU: 461 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 62 MB
- maxRAM: 2719 MB
- avgRAM: 334 MB

# of TraceLog entries: 8636

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 3
CPU values:

Provisioned CPU Cores: 2

- minCPU: 136 MHz
- maxCPU: 1532 MHz
- avgCPU: 179 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 245 MB
- maxRAM: 5034 MB
- avgRAM: 736 MB

# of TraceLog entries: 8640

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 4
CPU values:

Provisioned CPU Cores: 2

- minCPU: 66 MHz
- maxCPU: 1023 MHz
- avgCPU: 82 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 659 MB
- maxRAM: 4198 MB
- avgRAM: 1032 MB

# of TraceLog entries: 8928

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 5
CPU values:

Provisioned CPU Cores: 2

- minCPU: 61 MHz
- maxCPU: 755 MHz
- avgCPU: 71 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 605 MB
- maxRAM: 5482 MB
- avgRAM: 1008 MB

# of TraceLog entries: 8924

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 6
CPU values:

Provisioned CPU Cores: 1

- minCPU: 22 MHz
- maxCPU: 355 MHz
- avgCPU: 27 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 38 MB
- maxRAM: 1264 MB
- avgRAM: 94 MB

# of TraceLog entries: 8629

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 7
CPU values:

Provisioned CPU Cores: 1

- minCPU: 22 MHz
- maxCPU: 238 MHz
- avgCPU: 28 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 37 MB
- maxRAM: 1425 MB
- avgRAM: 95 MB

# of TraceLog entries: 8919

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 8
CPU values:

Provisioned CPU Cores: 4

- minCPU: 414 MHz
- maxCPU: 3859 MHz
- avgCPU: 526 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 261 MB
- maxRAM: 4100 MB
- avgRAM: 680 MB

# of TraceLog entries: 8922

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 9
CPU values:

Provisioned CPU Cores: 2

- minCPU: 70 MHz
- maxCPU: 1833 MHz
- avgCPU: 148 MHz

RAM values:

Provisioned RAM: 6 GB

- minRAM: 155 MB
- maxRAM: 3612 MB
- avgRAM: 386 MB

# of TraceLog entries: 8922

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 10
CPU values:

Provisioned CPU Cores: 4

- minCPU: 47 MHz
- maxCPU: 4479 MHz
- avgCPU: 307 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 147 MB
- maxRAM: 4280 MB
- avgRAM: 891 MB

# of TraceLog entries: 8922

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 11
CPU values:

Provisioned CPU Cores: 2

- minCPU: 44 MHz
- maxCPU: 4798 MHz
- avgCPU: 188 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 119 MB
- maxRAM: 2834 MB
- avgRAM: 806 MB

# of TraceLog entries: 8922

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 12
CPU values:

Provisioned CPU Cores: 2

- minCPU: 30 MHz
- maxCPU: 1121 MHz
- avgCPU: 73 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 76 MB
- maxRAM: 2787 MB
- avgRAM: 209 MB

# of TraceLog entries: 8919

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 13
CPU values:

Provisioned CPU Cores: 2

- minCPU: 31 MHz
- maxCPU: 1176 MHz
- avgCPU: 74 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 65 MB
- maxRAM: 2703 MB
- avgRAM: 204 MB

# of TraceLog entries: 8921

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 14
CPU values:

Provisioned CPU Cores: 8

- minCPU: 89 MHz
- maxCPU: 8632 MHz
- avgCPU: 928 MHz

RAM values:

Provisioned RAM: 16 GB

- minRAM: 719 MB
- maxRAM: 10669 MB
- avgRAM: 2883 MB

# of TraceLog entries: 8922

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 15
CPU values:

Provisioned CPU Cores: 8

- minCPU: 138 MHz
- maxCPU: 11215 MHz
- avgCPU: 1804 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 769 MB
- maxRAM: 7814 MB
- avgRAM: 3480 MB

# of TraceLog entries: 8640

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 16
CPU values:

Provisioned CPU Cores: 8

- minCPU: 22 MHz
- maxCPU: 13477 MHz
- avgCPU: 808 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 299 MB
- maxRAM: 7601 MB
- avgRAM: 2321 MB

# of TraceLog entries: 8928

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 17
CPU values:

Provisioned CPU Cores: 8

- minCPU: 21 MHz
- maxCPU: 4121 MHz
- avgCPU: 429 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 38 MB
- maxRAM: 6094 MB
- avgRAM: 1335 MB

# of TraceLog entries: 8928

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 18
CPU values:

Provisioned CPU Cores: 2

- minCPU: 16 MHz
- maxCPU: 1410 MHz
- avgCPU: 98 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 24 MB
- maxRAM: 1777 MB
- avgRAM: 526 MB

# of TraceLog entries: 8928

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 19
CPU values:

Provisioned CPU Cores: 2

- minCPU: 235 MHz
- maxCPU: 3171 MHz
- avgCPU: 663 MHz

RAM values:

Provisioned RAM: 16 GB

- minRAM: 2227 MB
- maxRAM: 13607 MB
- avgRAM: 5802 MB

# of TraceLog entries: 8928

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 20
CPU values:

Provisioned CPU Cores: 1

- minCPU: 11 MHz
- maxCPU: 2377 MHz
- avgCPU: 640 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 38 MB
- maxRAM: 1823 MB
- avgRAM: 752 MB

# of TraceLog entries: 8919

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 21
CPU values:

Provisioned CPU Cores: 2

- minCPU: 99 MHz
- maxCPU: 1503 MHz
- avgCPU: 210 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 261 MB
- maxRAM: 4652 MB
- avgRAM: 692 MB

# of TraceLog entries: 8346

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 22
CPU values:

Provisioned CPU Cores: 1

- minCPU: 12 MHz
- maxCPU: 723 MHz
- avgCPU: 27 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 24 MB
- maxRAM: 1603 MB
- avgRAM: 84 MB

# of TraceLog entries: 8918

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 23
CPU values:

Provisioned CPU Cores: 1

- minCPU: 13 MHz
- maxCPU: 711 MHz
- avgCPU: 28 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 2 MB
- maxRAM: 2875 MB
- avgRAM: 97 MB

# of TraceLog entries: 8918

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 24
CPU values:

Provisioned CPU Cores: 1

- minCPU: 33 MHz
- maxCPU: 55 MHz
- avgCPU: 38 MHz

RAM values:

```

#### Materna-Trace-3 (`/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-3/00-Trace-Info.txt`)

*(Total lines in file: 11516; excerpt: first 500 lines.)*

```text

TraceAnalysis:


Results for each Workload Trace:

Trace File: 1
CPU values:

Provisioned CPU Cores: 2

- minCPU: 95 MHz
- maxCPU: 5065 MHz
- avgCPU: 1867 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 46 MB
- maxRAM: 2138 MB
- avgRAM: 480 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 2
CPU values:

Provisioned CPU Cores: 2

- minCPU: 84 MHz
- maxCPU: 5199 MHz
- avgCPU: 905 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 43 MB
- maxRAM: 2577 MB
- avgRAM: 357 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 3
CPU values:

Provisioned CPU Cores: 2

- minCPU: 135 MHz
- maxCPU: 827 MHz
- avgCPU: 184 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 294 MB
- maxRAM: 3237 MB
- avgRAM: 749 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 4
CPU values:

Provisioned CPU Cores: 2

- minCPU: 67 MHz
- maxCPU: 1773 MHz
- avgCPU: 82 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 594 MB
- maxRAM: 4379 MB
- avgRAM: 1044 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 5
CPU values:

Provisioned CPU Cores: 2

- minCPU: 59 MHz
- maxCPU: 1559 MHz
- avgCPU: 69 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 573 MB
- maxRAM: 5345 MB
- avgRAM: 999 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 6
CPU values:

Provisioned CPU Cores: 1

- minCPU: 22 MHz
- maxCPU: 387 MHz
- avgCPU: 28 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 31 MB
- maxRAM: 1196 MB
- avgRAM: 96 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 7
CPU values:

Provisioned CPU Cores: 1

- minCPU: 21 MHz
- maxCPU: 380 MHz
- avgCPU: 27 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 31 MB
- maxRAM: 1388 MB
- avgRAM: 95 MB

# of TraceLog entries: 10366

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 8
CPU values:

Provisioned CPU Cores: 4

- minCPU: 433 MHz
- maxCPU: 3496 MHz
- avgCPU: 533 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 266 MB
- maxRAM: 5394 MB
- avgRAM: 652 MB

# of TraceLog entries: 10075

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 9
CPU values:

Provisioned CPU Cores: 2

- minCPU: 78 MHz
- maxCPU: 1905 MHz
- avgCPU: 227 MHz

RAM values:

Provisioned RAM: 6 GB

- minRAM: 151 MB
- maxRAM: 3256 MB
- avgRAM: 477 MB

# of TraceLog entries: 10363

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 10
CPU values:

Provisioned CPU Cores: 4

- minCPU: 57 MHz
- maxCPU: 4168 MHz
- avgCPU: 325 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 135 MB
- maxRAM: 8715 MB
- avgRAM: 963 MB

# of TraceLog entries: 9787

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 11
CPU values:

Provisioned CPU Cores: 2

- minCPU: 25 MHz
- maxCPU: 663 MHz
- avgCPU: 38 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 5 MB
- maxRAM: 474 MB
- avgRAM: 151 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 12
CPU values:

Provisioned CPU Cores: 2

- minCPU: 48 MHz
- maxCPU: 4593 MHz
- avgCPU: 365 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 163 MB
- maxRAM: 4221 MB
- avgRAM: 943 MB

# of TraceLog entries: 10080

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 13
CPU values:

Provisioned CPU Cores: 2

- minCPU: 22 MHz
- maxCPU: 1359 MHz
- avgCPU: 82 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 32 MB
- maxRAM: 2203 MB
- avgRAM: 194 MB

# of TraceLog entries: 10080

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 14
CPU values:

Provisioned CPU Cores: 2

- minCPU: 22 MHz
- maxCPU: 1184 MHz
- avgCPU: 82 MHz

RAM values:

Provisioned RAM: 4 GB

- minRAM: 35 MB
- maxRAM: 2506 MB
- avgRAM: 183 MB

# of TraceLog entries: 10366

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 15
CPU values:

Provisioned CPU Cores: 8

- minCPU: 174 MHz
- maxCPU: 12056 MHz
- avgCPU: 1028 MHz

RAM values:

Provisioned RAM: 16 GB

- minRAM: 916 MB
- maxRAM: 11565 MB
- avgRAM: 3014 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 16
CPU values:

Provisioned CPU Cores: 8

- minCPU: 1 MHz
- maxCPU: 12020 MHz
- avgCPU: 1310 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 81 MB
- maxRAM: 7868 MB
- avgRAM: 3075 MB

# of TraceLog entries: 10351

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 17
CPU values:

Provisioned CPU Cores: 8

- minCPU: 22 MHz
- maxCPU: 11054 MHz
- avgCPU: 1082 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 32 MB
- maxRAM: 7814 MB
- avgRAM: 2576 MB

# of TraceLog entries: 10356

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 18
CPU values:

Provisioned CPU Cores: 8

- minCPU: 21 MHz
- maxCPU: 4153 MHz
- avgCPU: 349 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 141 MB
- maxRAM: 5083 MB
- avgRAM: 1211 MB

# of TraceLog entries: 10356

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 19
CPU values:

Provisioned CPU Cores: 2

- minCPU: 1 MHz
- maxCPU: 1352 MHz
- avgCPU: 93 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 27 MB
- maxRAM: 1805 MB
- avgRAM: 406 MB

# of TraceLog entries: 10349

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 20
CPU values:

Provisioned CPU Cores: 2

- minCPU: 159 MHz
- maxCPU: 3260 MHz
- avgCPU: 734 MHz

RAM values:

Provisioned RAM: 16 GB

- minRAM: 1342 MB
- maxRAM: 14777 MB
- avgRAM: 6349 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 21
CPU values:

Provisioned CPU Cores: 2

- minCPU: 76 MHz
- maxCPU: 2325 MHz
- avgCPU: 170 MHz

RAM values:

Provisioned RAM: 8 GB

- minRAM: 278 MB
- maxRAM: 4745 MB
- avgRAM: 720 MB

# of TraceLog entries: 10363

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 22
CPU values:

Provisioned CPU Cores: 1

- minCPU: 11 MHz
- maxCPU: 2401 MHz
- avgCPU: 269 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 19 MB
- maxRAM: 1809 MB
- avgRAM: 520 MB

# of TraceLog entries: 10366

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 23
CPU values:

Provisioned CPU Cores: 1

- minCPU: 12 MHz
- maxCPU: 655 MHz
- avgCPU: 29 MHz

RAM values:

Provisioned RAM: 2 GB

- minRAM: 20 MB
- maxRAM: 1593 MB
- avgRAM: 85 MB

# of TraceLog entries: 10368

Time where VM is turned off: 0min. (0h., 0d.)

Trace File: 24
CPU values:

Provisioned CPU Cores: 1

- minCPU: 13 MHz
- maxCPU: 938 MHz
- avgCPU: 28 MHz

RAM values:

```

**Structure (one paragraph).** `00-Trace-Info.txt` is plain text, not a CSV. It opens with `TraceAnalysis:` and `Results for each Workload Trace:`, then repeats a template for each logical VM (“Trace File: *n*”). Each block reports **resource provisioning and observed ranges**: `Provisioned CPU Cores`, `Provisioned RAM`, min/max/avg CPU (MHz) and RAM (MB), `# of TraceLog entries`, and `Time where VM is turned off`. There are **no** application names, workload class labels, OS, hostname patterns, tenant IDs, or service categories—only numeric provisioning and utilization summaries plus trace length. This file is a **per-VM provisioning/utilization digest**, not an application-level similarity ground truth.

### 3. PER-VM CSV STRUCTURE

Exemplars: `501.csv` (Trace 1), `58.csv` (Trace 2), `238.csv` (Trace 3).

#### `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-1/501.csv`

- **Headers (verbatim one line):**

```
"Timestamp";"CPU cores";"CPU capacity provisioned [MHZ]";"CPU usage [MHZ]";"CPU usage [%]";"Memory capacity provisioned [KB]";"Memory usage [KB]";"Memory usage [%]";"Disk read throughput [KB/s]";"Disk write throughput [KB/s]";"Disk size [GB]";"Network received throughput [KB/s]";"Network transmitted throughput [KB/s]"
```

- **First 3 data rows (verbatim):**

```
"05.11.2015 00:00:00";"2";"0";"89";"1,95";"4194304";"477731";"11,39";"0";"9";"24";"21";"154"
"05.11.2015 00:05:00";"2";"0";"115";"2,51";"4194304";"628726";"14,99";"0";"12";"24";"29";"208"
"05.11.2015 00:10:00";"2";"0";"95";"2,08";"4194304";"570006";"13,59";"12";"12";"24";"19";"155"
```

- **Last 2 data rows (verbatim):**

```
"03.12.2015 23:50:00";"2";"0";"69";"1,51";"4194304";"561617";"13,39";"0";"5";"24";"15";"132"
"03.12.2015 23:55:00";"2";"0";"72";"1,58";"4194304";"589300";"14,05";"0";"7";"24";"16";"163"
```

- **Row count**: 8064; **Column count**: 13
- **Timestamp column**: `Timestamp`
- **Time range (parsed)**: 2015-11-05T00:00:00 → 2015-12-03T23:55:00
- **Median sampling interval (s)**: 300.0

#### `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-2/58.csv`

- **Headers (verbatim one line):**

```
"Timestamp";"CPU cores";"CPU capacity provisioned [MHZ]";"CPU usage [MHZ]";"CPU usage [%]";"Memory capacity provisioned [KB]";"Memory usage [KB]";"Memory usage [%]";"Disk read throughput [KB/s]";"Disk write throughput [KB/s]";"Disk size [GB]";"Network received throughput [KB/s]";"Network transmitted throughput [KB/s]"
```

- **First 3 data rows (verbatim):**

```
"04.12.2015 00:00:00";"4";"0";"49";"0,68";"8388608";"838022";"9,99";"0";"6";"108";"1";"1"
"04.12.2015 00:05:00";"4";"0";"43";"0,61";"8388608";"714709";"8,52";"0";"11";"108";"1";"1"
"04.12.2015 00:10:00";"4";"0";"40";"0,56";"8388608";"373293";"4,45";"0";"5";"108";"1";"1"
```

- **Last 2 data rows (verbatim):**

```
"03.01.2016 23:50:00";"4";"0";"40";"0,56";"8388608";"161061";"1,92";"0";"5";"108";"1";"1"
"03.01.2016 23:55:00";"4";"0";"41";"0,58";"8388608";"150156";"1,79";"0";"7";"108";"1";"1"
```

- **Row count**: 8928; **Column count**: 13
- **Timestamp column**: `Timestamp`
- **Time range (parsed)**: 2015-12-04T00:00:00 → 2016-01-03T23:55:00
- **Median sampling interval (s)**: 300.0

#### `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-3/238.csv`

- **Headers (verbatim one line):**

```
"Timestamp";"CPU cores";"CPU capacity provisioned [MHZ]";"CPU usage [MHZ]";"CPU usage [%]";"Memory capacity provisioned [KB]";"Memory usage [KB]";"Memory usage [%]";"Disk read throughput [KB/s]";"Disk write throughput [KB/s]";"Disk size [GB]";"Network received throughput [KB/s]";"Network transmitted throughput [KB/s]"
```

- **First 3 data rows (verbatim):**

```
"04.01.2016 00:00:00";"2";"0";"14";"0,33";"2097152";"90597";"4,32";"0";"0";"22";"0";"0"
"04.01.2016 00:05:00";"2";"0";"14";"0,33";"2097152";"52848";"2,52";"0";"0";"22";"0";"0"
"04.01.2016 00:10:00";"2";"0";"15";"0,37";"2097152";"83676";"3,99";"0";"0";"22";"0";"0"
```

- **Last 2 data rows (verbatim):**

```
"08.02.2016 23:50:00";"2";"0";"15";"0,37";"2097152";"78014";"3,72";"0";"0";"22";"0";"0"
"08.02.2016 23:55:00";"2";"0";"18";"0,45";"2097152";"75288";"3,59";"0";"1";"22";"0";"0"
```

- **Row count**: 10080; **Column count**: 13
- **Timestamp column**: `Timestamp`
- **Time range (parsed)**: 2016-01-04T00:00:00 → 2016-02-08T23:55:00
- **Median sampling interval (s)**: 300.0

### 4. CORPUS-WIDE STATISTICS

- **Total VM CSV files (all traces)**: 1592
- **Row counts per VM** (all files): min=82, q25=8352.0, median=8922.0, q75=10080.0, max=10368, mean=8794.104271, std=1629.067926
- **Column-structure consistency**: All 1592 files share the same 13 column names. Hash of **sorted** tuple of column names (SHA-256, first 16 hex chars): `3222adad977b9fce`; **1** distinct structure.
- **Time-series duration per VM (days, from first to last timestamp in each file): min=0.315972, q25=28.996528, median=30.996528, q75=35.996528, max=35.996528, mean=30.736281
- **Sampling interval**: Median of per-VM median step sizes is **300.0 s** (min/max across files: 300.0/300.0), consistent with **5-minute** sampling; the 5-minute assumption is **not** contradicted.

### 5. CPU AND MEMORY CHARACTERIZATION

**Columns in the VM CSVs:**

- **CPU (candidates)**: `CPU usage [MHZ]` (absolute MHz), `CPU usage [%]` (share of provisioned capacity).
- **Memory (candidates)**: `Memory usage [KB]` (absolute KB), `Memory usage [%]` (share of provisioned RAM).

**Best guess for DynAPM forecasting targets:**

- **`CPU usage [%]`** — Comparable across VMs with different core counts and MHz caps; better for cross-VM models than raw MHz.
- **`Memory usage [%]`** — Same rationale vs absolute KB.

### 6. DISTRIBUTIONAL SAMPLE (100 VMs, seed 42, stratified 33/33/34)

Per-VM stats used columns `CPU usage [%]` and `Memory usage [%]`. Skewness: sample third-moment skew (g1) on each VM series.

**Aggregated distributions of per-VM means (across the 100 VMs):** CPU — min=0.150347, p25=0.410550, median=1.137141, p75=4.902127, max=69.432982. Memory — min=0.055431, p25=1.302739, median=4.364085, p75=10.952087, max=52.307897.

**Per-VM std (same quantiles):** CPU — min=0.007954, p25=0.255634, median=1.113892, p75=2.754898, max=45.058261. Memory — min=0.202016, p25=1.273570, median=2.605982, p75=4.755079, max=17.424530.

**Per-VM skewness (same quantiles):** CPU — min=-3.930079, p25=3.614697, median=10.476110, p75=33.260912, max=85.231493. Memory — min=-4.572525, p25=1.848654, median=4.299115, p75=12.224681, max=45.242719.

### 7. PLOT GENERATION

**Not produced in this run.** The environment’s default `python3` has **no `matplotlib`** (and no `numpy`/`pandas`), so the four PNGs required in `dynapm/reports/task_00/` were not written. `dynapm/scripts/task00_characterize.py` contains the intended matplotlib code to generate `cpu_usage_examples.png`, `memory_usage_examples.png`, `per_vm_mean_distribution.png`, and `per_vm_skewness_distribution.png` once dependencies are available.

### 8. MISSING-DATA CHECK (100-VM sample)

- Fraction with any NaN in CPU or memory target columns: **0.0**
- Fraction with constant CPU (`CPU usage [%]`): **0.0**
- Fraction with constant memory (`Memory usage [%]`): **0.0**
- Fraction with fewer than 1000 timesteps: **0.0**

### 9. PYTHON ENVIRONMENT

- **Python**: 3.13.2 (`/opt/homebrew/bin/python3`).
- **Packages**: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `scipy`, `river`, `scikit-multiflow` — **all reported NOT INSTALLED** by import check (no versions).

### 10. DISK AND COMPUTE

- **Disk** (volume hosting the project, `df -h` on project path): 460 Gi total, **~28 Gi available** on `/System/Volumes/Data` (94% used on that volume).
- **CPU model**: Apple M3.
- **RAM**: 24 GB total.

## Assumptions

- **Decimal separator**: European comma in numeric fields is converted to dot for parsing.
- **Timestamps**: Parsed as `%d.%m.%Y %H:%M:%S` (day-first), matching values like `05.11.2015 00:00:00`.
- **Targets for items 6–8**: `CPU usage [%]` and `Memory usage [%]` as the operational forecasting targets (see §5).
- **Stratified 100-VM sample**: Counts 33/33/34 by trace to match relative corpus sizes.
- **Corpus statistics** (items 4, 6, 8): Computed with a **stdlib-only** pass identical in intent to `task00_characterize.py` because the mandated script could not import numpy/pandas; results are in `characterization_fallback.json`.

## Surprises

- **No scientific Python stack** on the default interpreter used for this task, causing immediate failure of `task00_characterize.py` and blocking PNG generation.
- **Trace-1** includes extra non-CSV files (e.g. `.bak`, extensionless) beyond VM CSVs and `00-Trace-Info.txt`.
- **Extreme positive skewness** on some VMs (e.g. per-VM CPU skew up to ~85) indicates heavy tails or many near-idle samples with rare spikes—relevant for drift/threshold design.

## Commands

*(Output truncated to at most 20 lines per command, per spec.)*

```text
$ mkdir -p .../dynapm/reports/task_00 .../dynapm/scripts
(exit 0, no output)
```

```text
$ find .../Materna-Trace-<n> -maxdepth 1 -type f -name '*.csv' | wc -l
Trace-1: 518
Trace-2: 527
Trace-3: 547
```

```text
$ python3 .../dynapm/scripts/task00_characterize.py
Traceback (most recent call last):
  File ".../task00_characterize.py", line 33, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

```text
$ python3 -m py_compile .../dynapm/scripts/task00_characterize.py
(exit 0)
```

```text
$ python3 --version
Python 3.13.2
```

```text
$ wc -l .../Materna-Trace-*/00-Trace-Info.txt
   10949 .../Materna-Trace-1/00-Trace-Info.txt
   11096 .../Materna-Trace-2/00-Trace-Info.txt
   11516 .../Materna-Trace-3/00-Trace-Info.txt
   33561 total
```

```text
$ python3 (import check for numpy, pandas, sklearn, xgboost, matplotlib, scipy, river, skmultiflow)
numpy NOT INSTALLED
pandas NOT INSTALLED
scikit-learn NOT INSTALLED
xgboost NOT INSTALLED
matplotlib NOT INSTALLED
scipy NOT INSTALLED
river NOT INSTALLED
skmultiflow/scikit_multiflow NOT INSTALLED
```

```text
$ df -h .../gwa-t-13-lstm-pipeline | tail -1
/dev/disk3s5   460Gi   410Gi    28Gi    94%    2.1M  299M    1%   /System/Volumes/Data
```

```text
$ sysctl -n machdep.cpu.brand_string
Apple M3
```

```text
$ sysctl -n hw.memsize
25769803776
```

```text
$ (stdlib corpus scan; ~58s runtime)
wrote .../dynapm/reports/task_00/characterization_fallback.json errors 0
```

## Deviations from the task specification

- **Packages missing**: The default `python3` cannot import `numpy` (and thus not `pandas`, `matplotlib`, or `scipy`). Per instructions, nothing was installed. The mandated script therefore did not run end-to-end; corpus and sample statistics were reproduced with an equivalent **stdlib-only** analysis whose results are stored in `dynapm/reports/task_00/characterization_fallback.json` (not produced by executing `task00_characterize.py`).
- **Plots (item 7)**: The four PNG files were **not** written because `matplotlib` is unavailable in this environment. The plotting logic remains in `task00_characterize.py` for a properly provisioned interpreter.
- **Column-structure hash in JSON**: `characterization_fallback.json` records hash `1b6ecbb551c24f13` (file column order). The spec asks for a hash of the **sorted** column-name tuple; that canonical hash is **`3222adad977b9fce`** (still exactly one distinct structure).

## Next-step suggestion

Provision a Python environment with at least **numpy, pandas, matplotlib, scipy** (and lock versions in `dynapm/requirements.txt` if desired), re-run `dynapm/scripts/task00_characterize.py` to emit `characterization_full.json` and the four PNGs under `dynapm/reports/task_00/`, then proceed to D1 similarity / drift design using `CPU usage [%]` and `Memory usage [%]` targets—recognizing that `00-Trace-Info.txt` offers provisioning statistics only, so application-level ground truth must come from elsewhere or be defined without those labels.

## Absolute paths — files read or written

**Written (created or modified):**

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/scripts/task00_characterize.py`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_00/characterization_fallback.json`
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/dynapm/reports/task_00_discovery.md`

**Read (data and inspection; no modifications):**

- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-1/` (all VM `*.csv`, `00-Trace-Info.txt`, other files enumerated in §1)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-2/` (same)
- `/Users/fgtek002/Downloads/gwa-t-13-lstm-pipeline/old/data/raw/Materna-Trace-3/` (same)

