# Experiment Summary: 20251229-ros-alth-mnist-cpu

Generated: 2025-12-29
Note: This summary reflects the initial mini-library run. A full-library, multi-layer OBL rerun is in progress.
Report: project/dashboard/data/report.json

## Executive summary

- MLP: OBL reached similar accuracy (+0.13 pp) but incurred large step-time and wall-time increases.
- CNN: OBL slightly reduced accuracy (-0.22 pp) with modest step-time (+4.7%) and wall-time (+3.5%) overheads.
- OBL variants increased parameter counts (MLP +161.6%, CNN +19.6%), affecting comparability.

## Key metrics (mean +/- std)

MLP baseline:
- Test accuracy: 97.79% +/- 0.08 pp
- Step time: 54.70 ms +/- 25.25
- Wall time: 753.3 s +/- 148.0
- Params: 203,530

MLP + OBL:
- Test accuracy: 97.92% +/- 0.12 pp
- Step time: 186.92 ms +/- 146.36
- Wall time: 989.3 s +/- 492.1
- Params: 532,522

CNN baseline:
- Test accuracy: 99.15% +/- 0.15 pp
- Step time: 361.54 ms +/- 342.12
- Wall time: 1127.4 s +/- 768.4
- Params: 421,642

CNN + OBL:
- Test accuracy: 98.93% +/- 0.09 pp
- Step time: 378.44 ms +/- 427.80
- Wall time: 1166.7 s +/- 1074.2
- Params: 504,234

## Comparisons (OBL - baseline)

MLP:
- Accuracy delta: +0.13 pp
- Step time delta: +241.7%
- Wall time delta: +31.3%
- Param delta: +161.6%

CNN:
- Accuracy delta: -0.22 pp
- Step time delta: +4.7%
- Wall time delta: +3.5%
- Param delta: +19.6%

## Observations

- The MLP OBL variant shows a substantial compute and parameter increase relative to baseline,
  which likely explains the step-time and wall-time overheads despite accuracy parity.
- The CNN OBL variant is closer in parameter count, with mild overhead but slightly lower accuracy.
- Variance in step-time and wall-time is high across seeds, suggesting CPU contention or
  variability in thread scheduling.

## Recommendations

- Rebalance hidden sizes to align parameter counts between baseline and OBL variants before
  making speed claims.
- Run with fixed CPU affinity or fewer background jobs to reduce timing variance.
- Consider reducing OBL operator counts for the MLP case (K=16 or smaller) to assess
  whether smaller OBLs deliver better efficiency.
