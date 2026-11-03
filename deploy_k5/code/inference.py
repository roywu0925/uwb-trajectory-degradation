"""SageMaker PyTorch serving handler for K5 (GRU, noise-augmented
training, 18,050 parameters, direct/no-physics-base variant).

Architecture is copied in full below, self-contained -- no dependency
on this project's own model.py, so the container only needs torch and
numpy (both already in the SageMaker PyTorch inference image).

Request/response schema (documented, not inferred -- see the chat
message this was produced for):

REQUEST (application/json):
{
  "instances": [
    {
      "hist": [[rel_x, rel_y, vx, vy], ...],   // one row per pre-throw
                                                // frame, oldest first;
                                                // the LAST row is the
                                                // anchor frame and its
                                                // rel_x/rel_y MUST be
                                                // (0.0, 0.0)
      "static": [is_defensive_coverage,         // one-hot, exactly one
                 is_targeted_receiver,           // of these two is 1.0
                 ball_land_rel_x,                // ball-land offset
                 ball_land_rel_y]                // from the anchor
    },
    ... // one object per player, any N >= 1
  ]
}

RESPONSE (application/json):
{
  "predictions": [
    {
      "model_out": [[dx_1, dy_1], ..., [dx_40, dy_40]],  // the network's
                    // own output at every one of the 40 horizon steps --
                    // this IS the full prediction for this "direct" (no
                    // physics base) variant: absolute predicted position
                    // at step t = anchor_xy + model_out[t-1]. Not added
                    // to anything server-side; the caller adds the
                    // anchor if they want absolute field coordinates.
      "primary_horizon_step": 19,
      "primary_horizon_seconds": 1.9,
      "primary_offset": [dx_19, dy_19]            // model_out[18], called
                                                    // out separately since
                                                    // 1.9s is this
                                                    // project's primary
                                                    // horizon
    },
    ...
  ]
}
"""

import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

HIST_DIM = 4
STATIC_DIM = 4
MAX_HORIZON = 40
PRIMARY_STEP = 19


class GRUResidualModel(nn.Module):
    def __init__(self, hidden: int = 64, mlp_hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(HIST_DIM, hidden, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + STATIC_DIM + 1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, hist_padded, hist_lengths, static_feat):
        packed = pack_padded_sequence(hist_padded, hist_lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        context = h_n[-1]

        B = context.shape[0]
        steps = torch.arange(1, MAX_HORIZON + 1, dtype=torch.float32,
                             device=context.device) / MAX_HORIZON
        steps = steps.view(1, MAX_HORIZON, 1).expand(B, MAX_HORIZON, 1)
        context_exp = context.unsqueeze(1).expand(B, MAX_HORIZON, self.hidden)
        static_exp = static_feat.unsqueeze(1).expand(B, MAX_HORIZON, STATIC_DIM)

        decoder_in = torch.cat([context_exp, static_exp, steps], dim=-1)
        return self.decoder(decoder_in)


# ------------------------------------------------------------ handlers

def model_fn(model_dir):
    model = GRUResidualModel(hidden=64)
    state_dict = torch.load(f"{model_dir}/model.pth", map_location="cpu",
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_num_threads(1)
    return model


def input_fn(request_body, content_type):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type: {content_type}")
    payload = json.loads(request_body)
    instances = payload["instances"]

    lengths = [len(inst["hist"]) for inst in instances]
    max_len = max(lengths)
    batch_size = len(instances)

    hist_padded = np.zeros((batch_size, max_len, HIST_DIM), dtype=np.float32)
    static_feat = np.zeros((batch_size, STATIC_DIM), dtype=np.float32)
    for i, inst in enumerate(instances):
        L = lengths[i]
        hist_padded[i, :L, :] = np.asarray(inst["hist"], dtype=np.float32)
        static_feat[i, :] = np.asarray(inst["static"], dtype=np.float32)

    return {
        "hist_padded": torch.from_numpy(hist_padded),
        "hist_lengths": torch.tensor(lengths, dtype=torch.int64),
        "static_feat": torch.from_numpy(static_feat),
    }


def predict_fn(input_data, model):
    with torch.no_grad():
        out = model(input_data["hist_padded"], input_data["hist_lengths"],
                   input_data["static_feat"])
    return out.numpy()   # (B, MAX_HORIZON, 2)


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError(f"unsupported accept type: {accept}")
    predictions = []
    for row in prediction:
        predictions.append({
            "model_out": row.tolist(),
            "primary_horizon_step": PRIMARY_STEP,
            "primary_horizon_seconds": PRIMARY_STEP / 10.0,
            "primary_offset": row[PRIMARY_STEP - 1].tolist(),
        })
    return json.dumps({"predictions": predictions})
