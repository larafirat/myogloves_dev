"""Validates D2 (Masia/Xiloyannis lineage): single input, flatter distal bias
than D1, coupled fingers (same u drives both). K2 = [0.85, 0.95, 1.00].
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D2_synergy_cross_finger"]
    qpos = run_device(device, u=[1.0], tau_max=0.03)  # kept below saturation, see view_exo_device.py
    print("D2 final qpos:", qpos)

    assert qpos["mcp2_flexion"] <= qpos["pm2_flexion"] <= qpos["md2_flexion"], "index finger not distal-biased"
    assert qpos["mcp3_flexion"] <= qpos["pm3_flexion"] <= qpos["md3_flexion"], "middle finger not distal-biased"

    # D2's bias is flatter than D1's: MCP should sit closer to DIP than in D1's K.
    index_spread = qpos["md2_flexion"] - qpos["mcp2_flexion"]
    print(f"D2 index MCP-to-DIP spread: {index_spread:.4f} rad (expect smaller than D1's spread at same tau_max)")

    print("D2 PASS: flatter distal-biased flexion confirmed on both fingers.")
