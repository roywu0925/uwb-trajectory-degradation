"""Tier 1 Task K: features, architecture, and loss for the learned
residual/direct model (TIER1_SPEC_4.md, amended by TIER1_SPEC_6.md).

Feature set per player, all anchor-relative:
  - history of relative position (x - anchor_x, y - anchor_y) and raw
    velocity (vx, vy), one row per pre-throw frame, variable length
  - the ball_land offset from the anchor (constant per track)
  - role, one-hot (only two roles occur among players-to-predict:
    'Defensive Coverage', 'Targeted Receiver' -- see config.py)

The anchor frame is the *last* row of `sample.history` for that player
(verified: frame_id and x/y match `sample.state` exactly), so the
relative-position feature at that last row is 0 by construction. That
is asserted in `build_examples`, not just claimed, per TIER1_SPEC_6.md's
"assert this the same way as the feature check".

Nothing from output_*.csv (i.e. `sample.truth`) touches a feature -- only
`sample.state` and `sample.history`, both pre-throw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import MAX_HORIZON

ROLES = ["Defensive Coverage", "Targeted Receiver"]
HIST_DIM = 4     # rel_x, rel_y, vx, vy
STATIC_DIM = len(ROLES) + 2   # role one-hot + ball_land rel offset


@dataclass
class Example:
    game_id: int
    play_id: int
    player_id: int
    role: str
    hist_feat: np.ndarray     # (L, HIST_DIM)
    static_feat: np.ndarray   # (STATIC_DIM,)
    anchor: np.ndarray        # (2,) absolute field x, y
    horizon: int              # actual valid horizon, <= MAX_HORIZON
    truth: np.ndarray         # (horizon, 2) absolute field coords


def _role_onehot(role: str) -> np.ndarray:
    out = np.zeros(len(ROLES), dtype=np.float32)
    out[ROLES.index(role)] = 1.0
    return out


def build_examples(samples: list) -> list[Example]:
    """One Example per player-to-predict, across every play in `samples`.

    Raises if any track's anchor-frame relative position is not ~0 --
    that would mean the anchor-frame assumption (last history row ==
    the state row) silently broke for some track.
    """
    examples = []
    max_abs_anchor_rel = 0.0
    for sample in samples:
        state = sample.state
        history_by_player = dict(tuple(
            sample.history.groupby("player_id", sort=False)))
        for i in range(len(state)):
            row = state.iloc[i]
            player_id = row["player_id"]
            anchor = np.array([row["x"], row["y"]], dtype=np.float64)

            hist = history_by_player.get(player_id)
            hist = hist.sort_values("frame_id")
            rel_x = hist["x"].to_numpy() - anchor[0]
            rel_y = hist["y"].to_numpy() - anchor[1]
            vx = hist["vx"].to_numpy()
            vy = hist["vy"].to_numpy()
            hist_feat = np.stack([rel_x, rel_y, vx, vy], axis=1).astype(np.float32)

            max_abs_anchor_rel = max(max_abs_anchor_rel,
                                     abs(rel_x[-1]), abs(rel_y[-1]))

            ball_rel = np.array([row["ball_land_x"] - anchor[0],
                                 row["ball_land_y"] - anchor[1]], dtype=np.float32)
            static_feat = np.concatenate([_role_onehot(row["role"]), ball_rel])

            examples.append(Example(
                game_id=sample.game_id, play_id=sample.play_id,
                player_id=player_id, role=row["role"],
                hist_feat=hist_feat, static_feat=static_feat,
                anchor=anchor.astype(np.float32),
                horizon=sample.horizon,
                truth=sample.truth[i].astype(np.float32),
            ))

    assert max_abs_anchor_rel < 1e-6, (
        f"anchor-frame relative position feature is not ~0 "
        f"(max |rel| = {max_abs_anchor_rel:.6f}) -- the assumption that "
        f"the last history row is the anchor frame has broken for some "
        f"track; features would silently carry a position offset instead "
        f"of a translation-invariant one")
    return examples


def build_examples_k4(samples: list, Q: np.ndarray) -> list[Example]:
    """K4: identical to build_examples, except the velocity channel
    (columns 2,3 of hist_feat) comes from the position-only Kalman
    filter's own per-frame velocity state (kalman.run_filter_sequence)
    instead of the raw finite-differenced vx,vy already in
    sample.history. rel_x/rel_y (position) are untouched -- only the
    velocity channel's SOURCE changes, per TIER1_SPEC_6.md's diagnostic
    finding that finite-differenced velocity is the amplification path
    the model inherited from Tier 1 Part 1's own signal-processing
    result. Same anchor-frame assertion as build_examples.
    """
    from kalman import run_filter_sequence

    examples = []
    max_abs_anchor_rel = 0.0
    for sample in samples:
        state = sample.state
        history_by_player = dict(tuple(
            sample.history.groupby("player_id", sort=False)))
        for i in range(len(state)):
            row = state.iloc[i]
            player_id = row["player_id"]
            anchor = np.array([row["x"], row["y"]], dtype=np.float64)

            hist = history_by_player.get(player_id)
            hist = hist.sort_values("frame_id")
            rel_x = hist["x"].to_numpy() - anchor[0]
            rel_y = hist["y"].to_numpy() - anchor[1]

            kalman_states = run_filter_sequence(hist, Q, use_velocity=False)
            vx = kalman_states[:, 2]
            vy = kalman_states[:, 3]
            hist_feat = np.stack([rel_x, rel_y, vx, vy], axis=1).astype(np.float32)

            max_abs_anchor_rel = max(max_abs_anchor_rel,
                                     abs(rel_x[-1]), abs(rel_y[-1]))

            ball_rel = np.array([row["ball_land_x"] - anchor[0],
                                 row["ball_land_y"] - anchor[1]], dtype=np.float32)
            static_feat = np.concatenate([_role_onehot(row["role"]), ball_rel])

            examples.append(Example(
                game_id=sample.game_id, play_id=sample.play_id,
                player_id=player_id, role=row["role"],
                hist_feat=hist_feat, static_feat=static_feat,
                anchor=anchor.astype(np.float32),
                horizon=sample.horizon,
                truth=sample.truth[i].astype(np.float32),
            ))

    assert max_abs_anchor_rel < 1e-6, (
        f"anchor-frame relative position feature is not ~0 "
        f"(max |rel| = {max_abs_anchor_rel:.6f})")
    return examples


def check_base_is_absolute(base_pred: np.ndarray, label: str) -> None:
    """The complement of the feature check above: base predictions must
    be in the same (absolute field) frame as `sample.truth`, not
    accidentally anchor-relative. A relative quantity here would sit
    within a few yards of 0; an absolute field coordinate never does
    (the field is 0-120 x 0-53.3 yd and plays don't happen at the
    boundary corner). TIER1_SPEC_6.md: "assert this the same way as the
    anchor-relative feature-check assertion."
    """
    med_abs = float(np.median(np.abs(base_pred)))
    assert med_abs > 5.0, (
        f"{label}: median |value| = {med_abs:.3f}, which looks "
        f"anchor-relative, not absolute-field -- base and truth must be "
        f"in the same frame before being added")


class GRUResidualModel(nn.Module):
    """Small GRU encoder over the (variable-length) pre-throw history,
    plus a decoder that produces one (dx, dy) output per horizon step
    from [final hidden state, static features, normalised step index].

    The decoder is applied to all MAX_HORIZON steps in one batched call
    (no per-step recurrence) -- this is still "a decoder producing
    per-step output" per TIER1_SPEC_4.md, just vectorised; it keeps the
    parameter count and the code small, and there is no reason a per-step
    autoregressive decoder would do better here since nothing about step
    t's output actually depends on step t-1's *predicted* output (the
    ground truth is never fed back at inference time either way).
    """

    def __init__(self, hidden: int = 64, mlp_hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(HIST_DIM, hidden, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + STATIC_DIM + 1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, hist_padded: torch.Tensor, hist_lengths: torch.Tensor,
               static_feat: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(hist_padded, hist_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        context = h_n[-1]                              # (B, hidden)

        B = context.shape[0]
        steps = torch.arange(1, MAX_HORIZON + 1, dtype=torch.float32,
                             device=context.device) / MAX_HORIZON
        steps = steps.view(1, MAX_HORIZON, 1).expand(B, MAX_HORIZON, 1)
        context_exp = context.unsqueeze(1).expand(B, MAX_HORIZON, self.hidden)
        static_exp = static_feat.unsqueeze(1).expand(B, MAX_HORIZON, STATIC_DIM)

        decoder_in = torch.cat([context_exp, static_exp, steps], dim=-1)
        return self.decoder(decoder_in)                 # (B, MAX_HORIZON, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def masked_mde_loss(pred: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean distance over unmasked steps.

    ||(base + model_out) - truth|| and ||model_out - (truth - base)|| are
    the same objective -- this function is called with pred = base +
    model_out and target = truth (final-prediction distance), which
    reads directly in the same yards-error units as every other table in
    this project, but is numerically identical to computing the residual
    target once and calling this on (model_out, truth - base).
    """
    dist = torch.linalg.norm(pred - target, dim=-1)     # (B, MAX_HORIZON)
    mask = mask.float()
    return (dist * mask).sum() / mask.sum().clamp(min=1.0)
