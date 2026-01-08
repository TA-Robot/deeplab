# SuperLoRA meaningful 32-condition plan (2026-01-08)
Design: 32-run orthogonal design (Hadamard 32) over 6 two-level factors.
Fixed: projection=fastfood, scope=all, group=1, init=kaiming, direction=none.
Factors (low/high):
- lr: 0.03 / 0.1
- momentum: 0.0 / 0.9
- rank: 8 / 32
- T (merge interval): 200 / 1000
- ws (warmstart): 0 / 100
- alpha_mult: 1 / 4 (alpha = rank * alpha_mult)

## Runs
01. 20260108-superlora-fastfood-meaningful32-01-lr0p03-m0p9-r8-T1000-ws0-a32 | lr=0.03, mom=0.9, rank=8, T=1000, ws=0, alpha=32
02. 20260108-superlora-fastfood-meaningful32-02-lr0p1-m0p0-r8-T1000-ws100-a8 | lr=0.1, mom=0.0, rank=8, T=1000, ws=100, alpha=8
03. 20260108-superlora-fastfood-meaningful32-03-lr0p03-m0p0-r32-T1000-ws0-a32 | lr=0.03, mom=0.0, rank=32, T=1000, ws=0, alpha=32
04. 20260108-superlora-fastfood-meaningful32-04-lr0p1-m0p9-r32-T200-ws0-a32 | lr=0.1, mom=0.9, rank=32, T=200, ws=0, alpha=32
05. 20260108-superlora-fastfood-meaningful32-05-lr0p03-m0p9-r8-T200-ws100-a8 | lr=0.03, mom=0.9, rank=8, T=200, ws=100, alpha=8
06. 20260108-superlora-fastfood-meaningful32-06-lr0p1-m0p0-r8-T200-ws0-a32 | lr=0.1, mom=0.0, rank=8, T=200, ws=0, alpha=32
07. 20260108-superlora-fastfood-meaningful32-07-lr0p03-m0p0-r32-T200-ws100-a128 | lr=0.03, mom=0.0, rank=32, T=200, ws=100, alpha=128
08. 20260108-superlora-fastfood-meaningful32-08-lr0p1-m0p9-r32-T1000-ws100-a128 | lr=0.1, mom=0.9, rank=32, T=1000, ws=100, alpha=128
09. 20260108-superlora-fastfood-meaningful32-09-lr0p03-m0p9-r8-T1000-ws0-a32 | lr=0.03, mom=0.9, rank=8, T=1000, ws=0, alpha=32
10. 20260108-superlora-fastfood-meaningful32-10-lr0p1-m0p0-r8-T1000-ws100-a8 | lr=0.1, mom=0.0, rank=8, T=1000, ws=100, alpha=8
11. 20260108-superlora-fastfood-meaningful32-11-lr0p03-m0p0-r32-T1000-ws0-a32 | lr=0.03, mom=0.0, rank=32, T=1000, ws=0, alpha=32
12. 20260108-superlora-fastfood-meaningful32-12-lr0p1-m0p9-r32-T200-ws0-a32 | lr=0.1, mom=0.9, rank=32, T=200, ws=0, alpha=32
13. 20260108-superlora-fastfood-meaningful32-13-lr0p03-m0p9-r8-T200-ws100-a8 | lr=0.03, mom=0.9, rank=8, T=200, ws=100, alpha=8
14. 20260108-superlora-fastfood-meaningful32-14-lr0p1-m0p0-r8-T200-ws0-a32 | lr=0.1, mom=0.0, rank=8, T=200, ws=0, alpha=32
15. 20260108-superlora-fastfood-meaningful32-15-lr0p03-m0p0-r32-T200-ws100-a128 | lr=0.03, mom=0.0, rank=32, T=200, ws=100, alpha=128
16. 20260108-superlora-fastfood-meaningful32-16-lr0p1-m0p9-r32-T1000-ws100-a128 | lr=0.1, mom=0.9, rank=32, T=1000, ws=100, alpha=128
17. 20260108-superlora-fastfood-meaningful32-17-lr0p03-m0p9-r8-T1000-ws0-a32 | lr=0.03, mom=0.9, rank=8, T=1000, ws=0, alpha=32
18. 20260108-superlora-fastfood-meaningful32-18-lr0p1-m0p0-r8-T1000-ws100-a8 | lr=0.1, mom=0.0, rank=8, T=1000, ws=100, alpha=8
19. 20260108-superlora-fastfood-meaningful32-19-lr0p03-m0p0-r32-T1000-ws0-a32 | lr=0.03, mom=0.0, rank=32, T=1000, ws=0, alpha=32
20. 20260108-superlora-fastfood-meaningful32-20-lr0p1-m0p9-r32-T200-ws0-a32 | lr=0.1, mom=0.9, rank=32, T=200, ws=0, alpha=32
21. 20260108-superlora-fastfood-meaningful32-21-lr0p03-m0p9-r8-T200-ws100-a8 | lr=0.03, mom=0.9, rank=8, T=200, ws=100, alpha=8
22. 20260108-superlora-fastfood-meaningful32-22-lr0p1-m0p0-r8-T200-ws0-a32 | lr=0.1, mom=0.0, rank=8, T=200, ws=0, alpha=32
23. 20260108-superlora-fastfood-meaningful32-23-lr0p03-m0p0-r32-T200-ws100-a128 | lr=0.03, mom=0.0, rank=32, T=200, ws=100, alpha=128
24. 20260108-superlora-fastfood-meaningful32-24-lr0p1-m0p9-r32-T1000-ws100-a128 | lr=0.1, mom=0.9, rank=32, T=1000, ws=100, alpha=128
25. 20260108-superlora-fastfood-meaningful32-25-lr0p03-m0p9-r8-T1000-ws0-a32 | lr=0.03, mom=0.9, rank=8, T=1000, ws=0, alpha=32
26. 20260108-superlora-fastfood-meaningful32-26-lr0p1-m0p0-r8-T1000-ws100-a8 | lr=0.1, mom=0.0, rank=8, T=1000, ws=100, alpha=8
27. 20260108-superlora-fastfood-meaningful32-27-lr0p03-m0p0-r32-T1000-ws0-a32 | lr=0.03, mom=0.0, rank=32, T=1000, ws=0, alpha=32
28. 20260108-superlora-fastfood-meaningful32-28-lr0p1-m0p9-r32-T200-ws0-a32 | lr=0.1, mom=0.9, rank=32, T=200, ws=0, alpha=32
29. 20260108-superlora-fastfood-meaningful32-29-lr0p03-m0p9-r8-T200-ws100-a8 | lr=0.03, mom=0.9, rank=8, T=200, ws=100, alpha=8
30. 20260108-superlora-fastfood-meaningful32-30-lr0p1-m0p0-r8-T200-ws0-a32 | lr=0.1, mom=0.0, rank=8, T=200, ws=0, alpha=32
31. 20260108-superlora-fastfood-meaningful32-31-lr0p03-m0p0-r32-T200-ws100-a128 | lr=0.03, mom=0.0, rank=32, T=200, ws=100, alpha=128
32. 20260108-superlora-fastfood-meaningful32-32-lr0p03-m0p0-r32-T200-ws100-a128 | lr=0.03, mom=0.0, rank=32, T=200, ws=100, alpha=128
