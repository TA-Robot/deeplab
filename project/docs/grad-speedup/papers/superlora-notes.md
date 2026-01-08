# SuperLoRA (BMVC 2024) — Notes

Source
- BMVC 2024 Paper 566 (PDF provided by user)

Summary
- SuperLoRA builds a low-rank adaptation by grouping layers, projecting group
  parameters into a shared higher-order tensor core, and optionally shuffling
  weights to align dimensions across layers.

Core formulation (paper sections)
- SuperLoRA concatenates all ΔW across layers and splits them into G groups.
- For each group, it applies a projection function F and then a tensorized low-rank
  update (LoRTA). The LoRTA form is described in Eq. (4), with reshaping after
  projected core multiplication.

Projection / shuffle
- The projection function F uses a Fastfood-style structured random projection
  (Figure 3) to map lower-dimensional parameters into the group tensor.
- Shuffling is treated as a simplified projection step: a random permutation Π is
  applied to each layer’s weight matrix to align dimensions across layers and
  distribute weights into an order suited for projection.

Notation / hyperparameters (from Tables 1–2)
- W_i: i-th layer weight; ΔW_i: its low-rank update.
- C_g: tensor core for group g.
- F: projection mapping from parameter space into tensor space.
- G: number of groups; K: number of splits; M: tensor order.
- ρ: projection ratio (controls compression).

Implementation mapping (ReLoRA integration)
- Treat SuperLoRA as an alternative adapter to ReLoRA:
  - Replace LoRA A/B with group-wise projection + tensor core update.
  - Keep ReLoRA merge/reset schedule (merge every T, reset adapter params).
- Default plan: implement group-wise projection + LoRTA update with optional shuffle.
- Projection choice: start with Fastfood (structured random) as in paper.

Open questions / risks
- Paper uses LoRTA; optional LoKr/LoTR variants are referenced elsewhere in the
  paper but need explicit mapping for Conv2d.
- Shuffle frequency and its interaction with merge/reset are not explicit; treat
  as a tunable interval.

