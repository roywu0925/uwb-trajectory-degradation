"""Gain test: measure d(output)/d(v_input) directly, forward-pass only,
clean data, fold-0 checkpoints. Confirms or refutes the "network's
learned gain on velocity exceeds the physical integration gain" theory
quantitatively rather than by argument.

Two variants of the perturbation, since they test slightly different
things:
  (a) perturb only the anchor (last history row)'s vx/vy -- the direct
      analog of constant_velocity's mechanism (uses only the anchor's
      velocity), so this is the fairest apples-to-apples comparison
      against the physical gain of 1.9 (displacement = v * 1.9s).
  (b) perturb every history row's vx/vy by the same delta -- closer to
      what real sigma_player noise actually does (every frame moves).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from config import OUTPUT_DIR, WEEKS
from cache_pipeline import build_dataset_cached
from kalman import calibrate_sigma_a, estimate_sigma_a
from model import GRUResidualModel, build_examples
from run_tier1_taskk import build_tensors, compute_base_arrays
from run_tier1_taskp import build_predictors_quadrature

PRIMARY_STEP = 19
DELTA = 0.01   # yd/s, small relative to typical speeds 3-8 yd/s


def main():
    samples, _ = build_dataset_cached(WEEKS)
    with open(OUTPUT_DIR / "folds.json") as f:
        fold_info = json.load(f)
    folds = fold_info["folds"]
    train_val_games = set(g for fold in folds for g in fold)
    train_val_samples = [s for s in samples if s.game_id in train_val_games]

    examples = build_examples(train_val_samples)
    tensors = build_tensors(examples)
    game_id_arr = tensors["game_id"]
    test_idx = np.where(np.isin(game_id_arr, folds[0]))[0]

    hist_padded = tensors["hist_padded"].numpy()
    hist_lengths = tensors["hist_lengths"].numpy()
    static_feat = tensors["static_feat"]
    anchor_np = tensors["anchor"].numpy()

    for variant in ("K1", "K3"):
        ckpt = torch.load(OUTPUT_DIR / "checkpoints" / f"{variant}_fold0.pt", weights_only=False)
        model = GRUResidualModel(hidden=ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        for perturb_mode in ("anchor_only", "all_frames"):
            gains_x, gains_y = [], []
            with torch.no_grad():
                for idx in test_idx:
                    L = hist_lengths[idx]
                    base_seq = hist_padded[idx:idx+1, :L, :].copy()   # (1, L, 4)

                    def forward(seq):
                        hp = torch.from_numpy(hist_padded[idx:idx+1].copy())
                        hp[:, :L, :] = torch.from_numpy(seq)
                        lengths = torch.tensor([L])
                        static = static_feat[idx:idx+1]
                        out = model(hp, lengths, static)   # (1, MAX_HORIZON, 2)
                        return out[0, PRIMARY_STEP - 1].numpy()

                    out0 = forward(base_seq[0])

                    seq_x = base_seq.copy()
                    seq_y = base_seq.copy()
                    if perturb_mode == "anchor_only":
                        seq_x[0, -1, 2] += DELTA
                        seq_y[0, -1, 3] += DELTA
                    else:
                        seq_x[0, :, 2] += DELTA
                        seq_y[0, :, 3] += DELTA

                    out_x = forward(seq_x[0])
                    out_y = forward(seq_y[0])

                    gains_x.append((out_x[0] - out0[0]) / DELTA)
                    gains_y.append((out_y[1] - out0[1]) / DELTA)

            gains_x = np.array(gains_x)
            gains_y = np.array(gains_y)
            print(f"{variant} [{perturb_mode}]: "
                 f"d(out_x)/d(vx): mean={gains_x.mean():.3f} median={np.median(gains_x):.3f} "
                 f"std={gains_x.std():.3f} | "
                 f"d(out_y)/d(vy): mean={gains_y.mean():.3f} median={np.median(gains_y):.3f} "
                 f"std={gains_y.std():.3f}")
            print(f"  (physical integration gain at 1.9s horizon = 1.9)")


if __name__ == "__main__":
    main()
