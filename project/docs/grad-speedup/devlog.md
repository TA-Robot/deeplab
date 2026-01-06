# Grad-Speedup Devlog

2026-01-05
- Added grad-speedup isolated codebase and docs; committed as 3b7f97d.
- Added critical-path doc and PM plan updates.
- Started baseline/silver runs; outputs landed under /workspace/deeplab/runs/grad-speedup (needs migration).
- Added PM agent rules to AGENTS.md and spawned parallel sub-agents for paper audit and module implementations.
- Migrated baseline run artifacts into project/runs/grad-speedup and updated PM status.
- Split grad-speedup tickets into project/docs/grad-speedup/tickets for experiment isolation.
- Updated PM docs (plan/critical-path/experiment-log) to reflect current blockers and next steps.
- Added local paper pack under project/docs/grad-speedup/papers and updated method-conformance with paper-accurate equations.
- Restarted sub-agents with updated tickets and moved active tickets to project/docs/grad-speedup/tickets/running.
- Added dashboard spec and queued dashboard implementation lane.
- Launched dashboard sub-agents (data + UI) and moved tickets to running.
- Completed dashboard data layer and Streamlit UI; tickets moved to done.

2026-01-06
- Corrected SAGD paper reference (arXiv:2509.14969) and updated step-control implementation to Variant III.
- Added SAGD delta flag (step_sagd_delta) and aligned configs/docs to use λ0 via optimizer LR.
- Ran SAGD GPU smoke (small-cnn, 1 epoch) to validate new step-control path.
- Ran step-control GPU smoke suite (small-cnn, 1 epoch each) across baseline/l0l1/sps/sps-momentum/adaptive-backtracking/sagd/silver.
- Ran direction GPU smoke suite (small-cnn, 1 epoch each) across none/diag-precond/shampoo/soap/sophia/muon.
- Ran Stage 3 pairwise combos (small-cnn, 1 epoch each) for l0l1/sps/silver × directions.
- Noted train metrics issue (samples=0/global_step=0) in recent runs; triage ticket opened.
- Fixed indentation bug in train loop that zeroed metrics for non-SAGD runs; verified metrics via GPU smoke.
- Re-ran Stage1/2/3 sweeps with metrics fix (step-control-v2, direction-v2, stage3-v2).
- Ranked Stage3 v2; best pair at epoch1 test acc: l0l1 + soap.
- Ran Stage4 sweep for l0l1 + soap; best variant at epoch1 test acc: anderson enabled, clip=none.
- Started Stage5 promotion runs on resnet18 (baseline + l0l1+soap+anderson); reran seed-2 for winner and regenerated summary.
- Stage5 preliminary results: baseline test acc 0.675±0.040 (n=3), l0l1+soap+anderson 0.847±0.011 (n=3) at epoch 5.
- Completed Stage6 baseline run (resnet18, 20 epochs, seeds 0/1/2).
- Resumed Stage6 winner run (l0l1+soap+anderson) for seeds 1/2 after earlier timeout; queued ablations (l0l1-only, soap-only, l0l1+soap).
- Paused new experiments pending step-based control (max_steps=14,000, eval_interval_steps=1000) and dashboard metric updates.
- Implemented dashboard data cleanup (none/None normalization) and elapsed time columns for steps/epochs.
- Updated dashboard UI for configurable hues/labels, bar sorting/top‑K, accuracy vs time, and plot‑selection to Run Detail.
- Implemented step-based training control in runner (max_steps + eval_interval_steps) with step-triggered eval logging.
- Cleared pre-step-based run artifacts under project/runs/grad-speedup to avoid mixed baselines.
- Ran step-based smoke (CPU) with max_steps=10, eval_interval_steps=5; verified test eval entries at steps 5 and 10.
- Started step-based baseline run on resnet18 (max_steps=14,000, eval_interval_steps=1000, seeds 0/1/2).
- Queued step-based winner run to auto-start after baseline (l0l1+soap+anderson, max_steps=14,000).
- Queued step-based ablations to auto-start after winner (l0l1-only, soap-only, l0l1+soap).
- Baseline completed; winner now running. Added post-ablation plan to mirror small-cnn sweeps on resnet18.
- Added queue runner scripts to allow sequential, appendable experiment execution without manual waiting; dashboard now surfaces run status/progress.
- Dashboard now ingests queue.txt so queued runs appear with status=queued before execution.

2026-01-06
- Realigned grad-speedup plan/system/combination docs to the CIFAR-10 spec (72-condition grid).
- Marked SOAP-heavy experiments as legacy; base grid now excludes direction/preconditioning track.
- Added spec-aligned CIFAR-10 implementation doc and refreshed critical path.
- Prepared new tickets for EoSS, grid generator/queue, and Layerwise GN.
