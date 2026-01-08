# SuperLoRA projection diagnosis plan (2026-01-08)
Goal: isolate impact of projection choice vs alpha scaling.

Fixed: resnet18, relu, scope=all, group=1, shuffle=off, max_steps=2000, seed=0.
Base hyperparams: lr=0.03, momentum=0.9, rank=32, T=1000, ws=100.
Factors: projection {none,fixed,learned,fastfood} × alpha_mult {1,4} (alpha = rank * alpha_mult).

## Runs
01. 20260108-superlora-projdiag-01-projnone-a32-r32-T1000-ws100-lr0p03-m0p9 | projection=none, alpha=32 (mult=1)
02. 20260108-superlora-projdiag-02-projnone-a128-r32-T1000-ws100-lr0p03-m0p9 | projection=none, alpha=128 (mult=4)
03. 20260108-superlora-projdiag-03-projfixed-a32-r32-T1000-ws100-lr0p03-m0p9 | projection=fixed, alpha=32 (mult=1)
04. 20260108-superlora-projdiag-04-projfixed-a128-r32-T1000-ws100-lr0p03-m0p9 | projection=fixed, alpha=128 (mult=4)
05. 20260108-superlora-projdiag-05-projlearned-a32-r32-T1000-ws100-lr0p03-m0p9 | projection=learned, alpha=32 (mult=1)
06. 20260108-superlora-projdiag-06-projlearned-a128-r32-T1000-ws100-lr0p03-m0p9 | projection=learned, alpha=128 (mult=4)
07. 20260108-superlora-projdiag-07-projfastfood-a32-r32-T1000-ws100-lr0p03-m0p9 | projection=fastfood, alpha=32 (mult=1)
08. 20260108-superlora-projdiag-08-projfastfood-a128-r32-T1000-ws100-lr0p03-m0p9 | projection=fastfood, alpha=128 (mult=4)
