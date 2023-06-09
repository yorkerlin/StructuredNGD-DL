import numpy as np
from .reglayers import AnyNet
import torch.nn as nn
import torch
#adapted from https://github.com/yhhhli/RegNet-Pytorch

__all__ = [
    'regnet_1600m',
    'regnet_3200m',
]


regnet_1600M_config = {'WA': 34.01, 'W0': 80, 'WM': 2.25, 'DEPTH': 18, 'GROUP_W': 24, 'BOT_MUL': 1}
regnet_3200M_config = {'WA': 26.31, 'W0': 88, 'WM': 2.25, 'DEPTH': 25, 'GROUP_W': 48, 'BOT_MUL': 1}

def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))       # ks = [0,1,2...,3...]
    ws = w_0 * np.power(w_m, ks)                             # float channel for 4 stages
    ws = np.round(np.divide(ws, q)) * q                      # make it divisible by 8
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    # ws: width list, num_stages: 4, max_stage: 4.0, wscont: float before round width
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, cfg, num_classes=10):
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH']
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [cfg['GROUP_W'] for _ in range(num_s)]
        bms = [cfg['BOT_MUL'] for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # stride for cifar is set to 1,2,2,2
        ss = [1 if i==0 else 2 for i in range(num_s)]
        # Use SE for RegNetY
        se_r = None
        # Construct the model
        STEM_W = 32
        kwargs = {
            "stem_w": STEM_W,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": num_classes,
        }
        super(RegNet, self).__init__(**kwargs)


def regnet_1600m(num_classes, **kwargs):
    model = RegNet(num_classes=num_classes, cfg=regnet_1600M_config)
    return model

def regnet_3200m(num_classes, **kwargs):
    model = RegNet(num_classes=num_classes, cfg=regnet_3200M_config)
    return model


# if __name__ == "__main__":
    # # img = torch.ones([8, 3, 32, 32], device='cuda')
    # img = torch.ones([8, 3, 32, 32], device='cuda')
    # model = regnet_200m(num_classes=100).to('cuda')
    # # model = PyramidNet(depth=101, alpha=64, num_classes=10, bottleneck=True, small_net=False)
    # model.train()
    # out_img = model(img)
    # print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]

