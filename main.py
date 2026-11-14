#!/usr/bin/env python3
"""Quantile forecaster for the polymarket attribution family."""
import json
import math
import os
import sys

TAUS = (0.05, 0.25, 0.50, 0.75, 0.95)
Z = (-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536)

K_FLOOR = 0.00050
K_CAP = 0.00120
K_MULT = 1.5
GRAVITY = 0.45
SIG_WINDOW = 120


def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def sigma_per_min(prices, window=SIG_WINDOW):
    tail = prices[-window:] if len(prices) > window else prices
    if len(tail) < 2:
        return K_FLOOR
    total = 0.0
    for i in range(1, len(tail)):
        total += abs(tail[i] - tail[i - 1])
    mean_abs = total / (len(tail) - 1)
    return mean_abs * 1.2533


def forecast_market(prices, horizon):
    x = prices[-1] if prices else 0.5
    k = sigma_per_min(prices) * K_MULT
    if k < K_FLOOR:
        k = K_FLOOR
    elif k > K_CAP:
        k = K_CAP
    rows = []
    for h in range(1, horizon + 1):
        f = float(h) / float(horizon)
        if x <= 0.5:
            centre = x * (1.0 - GRAVITY * f)
        else:
            centre = x + (1.0 - x) * GRAVITY * f
        spread = k * math.sqrt(h)
        row = [clamp01(centre + z * spread) for z in Z]
        rows.append([round(v, 6) for v in row])
    return rows


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        horizon = int(case.get("horizon_minutes") or 240)
        forecasts = []
        for market in case.get("markets") or []:
            rows = forecast_market(market.get("prices") or [], horizon)
            forecasts.append({"id": market.get("id"), "q": rows})
        path = os.path.join(out_dir, case_id + ".out")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"forecasts": forecasts}, fh, separators=(",", ":"))
    sys.stdout.write("wrote %d case(s)\n" % len(names))


if __name__ == "__main__":
    main()
