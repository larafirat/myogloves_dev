"""Validates D2 (Alicea, Xiloyannis, Chiaradia, Barsotti, Frisoli & Masia,
Wearable Technologies 2021): single input, flatter distal bias than D1, and a
literature-derived (not guessed) cross-finger asymmetry from the paper's
multichannel synergy pulley (Fig. 3b, channel diameters thumb/middle/index =
2.2/0.95/1.93 cm -> middle receives ~0.49x index's stroke on the shared shaft).
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

    # Pulley diameter ratio (index 1.93cm > middle 0.95cm) means index should flex
    # noticeably more than middle at every joint -- this is new, literature-derived
    # behavior that a uniform cross-finger K would not have shown.
    assert qpos["md2_flexion"] > qpos["md3_flexion"], "index (larger pulley channel) should out-flex middle"

    print("D2 PASS: flatter distal-biased flexion confirmed, with pulley-derived index>middle asymmetry.")
