# Grad-Speedup PM Status

Date: 2026-01-06
Owner: PM

Current state
- Critical-path doc: project/docs/grad-speedup/critical-path.md
- Method conformance: project/docs/grad-speedup/method-conformance.md (paper audit in progress)
- Tickets: project/docs/grad-speedup/tickets/ready/
- Running tickets: project/docs/grad-speedup/tickets/running/
- Paper pack: project/docs/grad-speedup/papers/README.md
- Additional papers downloaded for SPS/SAGD/LinBreg/Anderson/Shampoo (SAGD corrected to arXiv:2509.14969).
- Dashboard spec: project/docs/grad-speedup/dashboard-spec.md
- Parallel lanes: audit + step-control + direction methods + GGNC/Anderson + LinBreg
- Step-control: SAGD corrected to Variant III (arXiv:2509.14969), smoke verified.
- Combination plan: staged sweep documented in project/docs/grad-speedup/combination-plan.md
- Resolved: train metrics/logs now non-zero after indentation fix in train loop; prior rankings invalidated.
- Parallelization: Stage4 runner prep will proceed in parallel with metrics triage.

Sub-agent lanes (running / resumed)
- triage-paper-audit: populate method-conformance with paper-accurate rules (papers now local)

Sub-agent lanes (completed)
- implementer-stage6-ablations: superseded (epoch-based ablations deprecated; step-based ticket pending)
- implementer-dashboard-data: normalized categories + added elapsed time columns in dashboard data
- implementer-dashboard-ui: improved labels/colors/sorting/learning-curve UX + plot selection
- implementer-step-based-runner: added max_steps + eval_interval_steps; step-based eval/logging
- impl-step-control: paper-accurate step-control methods (papers now local)
- impl-soap-shampoo: direction methods (SOAP/Shampoo) (papers now local)
- impl-sophia-muon: direction methods (Sophia/Muon) (papers now local)
- impl-ggnc-anderson: stability/outer accel (GGNC paper now local)
- impl-linbreg: sparsity (awaits LinBreg paper confirmation)
- dashboard-data: data ingestion + derived metrics (done)
- dashboard-ui: Streamlit UI implementation (done)

Experiments (baseline smoke complete)
- 20260105-grad-speedup-resnet18-baseline (completed; outputs in project/runs/grad-speedup)
- 20260105-grad-speedup-smallcnn-baseline (completed; outputs in project/runs/grad-speedup)
- 20260105-grad-speedup-smallcnn-silver (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-cifar10-sagd-smoke-smallcnn-gpu (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-step-control-smoke-suite (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-direction-smoke-suite (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-stage3-pairwise (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-stepcontrol-v2 (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-direction-v2 (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-stage3-v2 (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-stage4 (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-stage5-resnet18 (completed; outputs in project/runs/grad-speedup)
- 20260106-grad-speedup-stage6-resnet18-ep20 (in progress; outputs in project/runs/grad-speedup)

PM actions queued
- Restart sub-agents with paper pack and updated tickets.
- Launch dashboard sub-agents and track deliverables.
- Update experiment-log.md with completed baseline status and artifact location.
- Update method-conformance once audit completes; promote remaining core methods to "audited".
- Remove or ignore empty /workspace/deeplab/runs/grad-speedup to avoid confusion.
- Run 1-epoch smoke for SAGD (extra gradient pass) and update experiment-log.md.
- Start combo Stage 2 once direction methods are merged.
- Stage3 v2 top pair (epoch1 test acc): l0l1 + soap (0.6342).
- Stage4 top variant (epoch1 test acc): l0l1 + soap + anderson (clip=none) acc 0.6883.
- Next: promote Stage4 winner(s) to multi-seed + longer epochs on resnet18.
  - baseline: 20260106-grad-speedup-stage5-baseline-resnet18-seeds012 (completed)
  - winner: 20260106-grad-speedup-stage5-l0l1-soap-anderson-resnet18-seeds012 (seed2 rerun completed; summary regenerated)
  - preliminary test acc @ epoch5 (mean±std, n=3): baseline 0.675±0.040; l0l1+soap+anderson 0.847±0.011
- Stage6 plan: 20-epoch promotion + ablations (baseline, l0l1-only, soap-only, l0l1+soap, winner).
- Stage6 status: baseline done; winner seed-0 complete; seeds 1/2 running.
- Next runs queued: l0l1-only, soap-only, l0l1+soap (no anderson) ablations on resnet18 (20 epochs, seeds 0/1/2).
- Experiments paused pending step-based control + eval cadence changes (potential re-run).
- Step-based defaults confirmed: max_steps=14,000, eval_interval_steps=1000 (epoch-based runs likely re-run).
- Run artifacts reset: project/runs/grad-speedup cleared to remove epoch-based results.
- Step-based smoke completed: 20260106-grad-speedup-step-smoke (max_steps=10, eval_interval_steps=5).
- Step-based baseline completed: 20260106-grad-speedup-step-baseline-resnet18-maxsteps14000-seeds012.
- Winner running: 20260106-grad-speedup-step-l0l1-soap-anderson-resnet18-maxsteps14000-seeds012.
- Step-based ablations queued to auto-start after winner:
  - 20260106-grad-speedup-step-l0l1-only-resnet18-maxsteps14000-seeds012
  - 20260106-grad-speedup-step-soap-only-resnet18-maxsteps14000-seeds012
  - 20260106-grad-speedup-step-l0l1-soap-resnet18-maxsteps14000-seeds012
- Next sweep plan: ResNet18 Stage1r–Stage4r at max_steps=7,000 (1 seed) before multi-seed promotion.
- Queue runner added for sequential execution; dashboard shows run status/progress.
- Dashboard now reads queue.txt to display queued runs before they start.
- Launch sub-agents: (1) triage metrics bug, (2) Stage4 runner prep.

Risks / blockers
- Paper-accurate definitions are a hard gate; implementations cannot be accepted until audit completes.
- Remaining paper gaps: SPS/SPS+Momentum, LinBreg, Armijo line search.
- No multi-seed baselines yet; only single-seed smoke runs completed.
