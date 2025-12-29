# AGENTS.md

PM directives for the deep learning speedup experiment program.

## Mission

Improve training and inference speed for target models without unacceptable
quality regressions. The focus is evidence-driven performance work, not new
model features.

## Scope and non-goals

- In scope: performance experiments, benchmarking, profiling, tuning, and
  reproducible reporting.
- Out of scope: product feature work, model architecture changes that are not
  motivated by speed, or experiments without a measurable baseline.

## Roles and ownership

- PM: owns prioritization, budget, and acceptance criteria.
- TL: owns technical direction, baselines, and experiment reviews.
- Experimenter: designs and runs experiments, writes reports.
- Infra: owns hardware, drivers, and runtime stability.
- Reviewer: validates results and checks for regressions.

## Operating principles

- Start from a measured baseline and quantify deltas.
- Change one major variable at a time unless explicitly approved.
- Prefer minimal diffs that can be reverted quickly.
- Optimize for reproducibility and traceability.
- Record compute cost and time-to-result.

## Directory and artifact rules

Keep the repo clean and make artifacts discoverable.

- Code and configs live under `project/`.
- Experiment notes live under `project/docs/`.
- Large artifacts and logs go under `project/runs/` and should not be committed.
- If a new directory is needed, document it in `project/docs/decisions.md`.

## Experiment lifecycle

1) Proposal: define hypothesis, baseline, metric, and success threshold.
2) Design: specify changes, dataset version, seeds, and expected cost.
3) Execution: run, collect metrics, and store artifacts.
4) Analysis: compare to baseline, check quality regressions.
5) Decision: accept, reject, or iterate. Record in the log.

## Experiment registration

Every experiment must have a short ID and a brief.

ID format: `YYYYMMDD-short-name`

Required brief fields:
- Hypothesis
- Baseline (commit, config, dataset version)
- Primary metric (e.g., step time, throughput, latency)
- Quality guardrail (e.g., accuracy, loss, BLEU, WER)
- Acceptance criteria (e.g., >= 8% speedup, <= 0.2% quality drop)
- Resource budget (GPU hours, max wall clock)

## Metrics and evaluation

- Always report primary speed metric and a quality guardrail.
- Use consistent batch sizes and input shapes for comparisons.
- If variance is high, run at least 3 trials and report mean and std.
- Record hardware details (GPU model, driver, CUDA, cuDNN).

## Reproducibility requirements

- Record random seeds and any determinism flags.
- Store configs and exact CLI args with the run artifacts.
- Capture commit hashes for all repos involved.
- Note environment changes (driver, kernel, library versions).

## Data handling

- Use versioned datasets and document preprocessing steps.
- Avoid mixing training and eval data. Validate splits.
- If data is sensitive, follow the stricter access policy and redaction rules.

## Resource and scheduling rules

- Reserve large runs in advance and log expected GPU hours.
- Cancel or stop runs that exceed budget or violate guardrails.
- Do not preempt shared resources without notifying Infra.

## Reporting and decisions

- Summarize outcomes in `project/docs/experiment-log.md`.
- Add decisions to `project/docs/decisions.md` with rationale.
- If a change is adopted, create a follow-up task to integrate it.

## Change management

- Every performance change must include a before/after result.
- Avoid bundling unrelated refactors with performance work.
- Ask for review before landing changes that affect correctness.

## Definition of done

- Experiment brief completed.
- Baseline and result reported with artifacts.
- Decision logged with next steps.
- No unacknowledged regressions in quality metrics.

## Templates

Experiment brief:

```
ID:
Owner:
Date:
Hypothesis:
Baseline:
Change:
Primary metric:
Quality guardrail:
Acceptance criteria:
Budget:
Notes:
```

Result summary:

```
ID:
Baseline:
Result:
Delta:
Quality:
Hardware:
Runtime:
Artifacts:
Decision:
```

## Sub-agent note

If a sub-agent is used for work under `project/`, also follow the additional
rules in `project/AGENTS.md`.
