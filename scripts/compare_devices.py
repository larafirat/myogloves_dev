"""Cross-device comparison: runs all D1-D4 variants (design-intent and
MyoHand-calibrated) under a full-drive input and reports each one's resulting
per-joint flexion and bias shape side by side, so the literature-vs-anatomy
shift documented individually in tests/test_d*.py can be read in one table.

Run from the repo root: python myogloves_dev/scripts/compare_devices.py
"""

import mujoco

from exo_devices import DEVICES, ExoApplicator, JOINT_NAMES

MODEL_PATH = "myogloves_dev/models/myohand_exoglove.xml"
STEPS = 1000
JOINT_LIMIT_RAD = 1.5708  # 90 deg, MyoHand's mcp/pip/dip joint range max
SAT_TOL = 0.1

# Display order: design-intent immediately followed by its calibrated pair.
ORDER = [
    "D1_underactuated_distal", "D1_underactuated_distal_calibrated",
    "D2_synergy_cross_finger", "D2_synergy_cross_finger_calibrated",
    "D3_uniform_single_dof", "D3_uniform_single_dof_calibrated",
    "D4_hybrid_per_finger", "D4_hybrid_per_finger_calibrated",
]

CITATIONS = {
    "D1": "Zhao et al. 2025, Biomimetic Intelligence and Robotics 5, 100248",
    "D2": "Alicea et al. 2021, Wearable Technologies 2, e4",
    "D3": "Thimabut et al. 2022, Rehabilitation Research and Practice, Art. 3738219",
    "D4": "Gerez, Gao, Dwivedi & Liarokapis 2020, IEEE Access 8, 173345-173358",
}


def run(device):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    applicator = ExoApplicator(model)
    if not device.tau_max_is_calibrated:
        device.tau_max[:] = 0.03  # same below-saturation convention as view_exo_device.py
    u = [1.0] * device.n_inputs
    for _ in range(STEPS):
        applicator.apply(data, device, u=u)
        mujoco.mj_step(model, data)
    qadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    return dict(zip(JOINT_NAMES, (round(data.qpos[a], 4) for a in qadr)))


TIE = 0.03  # rad; joints within this of each other are treated as tied, not ordered


def classify(mcp, pip, dip):
    if min(mcp, pip, dip) >= JOINT_LIMIT_RAD - SAT_TOL:
        return "SATURATED"
    if min(mcp, pip, dip) >= JOINT_LIMIT_RAD - 0.3:
        return "near-saturated"
    if abs(mcp - pip) <= TIE and mcp > dip + TIE and pip > dip + TIE:
        return "proximal-biased (MCP=PIP)"
    if dip > pip + TIE and pip > mcp + TIE:
        return "distal-biased"
    if mcp > pip + TIE and pip > dip + TIE:
        return "proximal-biased"
    if pip > mcp + TIE and pip > dip + TIE:
        return "PIP-peaked"
    return "mixed"


def main():
    rows = []
    for name in ORDER:
        device = DEVICES[name]
        qpos = run(device)
        index_vals = (qpos["mcp2_flexion"], qpos["pm2_flexion"], qpos["md2_flexion"])
        middle_vals = (qpos["mcp3_flexion"], qpos["pm3_flexion"], qpos["md3_flexion"])
        rows.append({
            "name": name,
            "calibrated": device.tau_max_is_calibrated,
            "tau_max": list(device.tau_max),
            "index": index_vals,
            "index_bias": classify(*index_vals),
            "middle": middle_vals,
            "middle_bias": classify(*middle_vals),
        })

    cols = ("Device", "tau_max scale", "Index (MCP/PIP/DIP)", "Index bias", "Middle (MCP/PIP/DIP)", "Middle bias")
    widths = (34, 15, 22, 16, 22, 16)
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        idx_str = "/".join(f"{v:.2f}" for v in r["index"])
        mid_str = "/".join(f"{v:.2f}" for v in r["middle"])
        tau_str = ("real: " if r["calibrated"] else "placeholder: ") + ",".join(f"{t:.4f}" for t in r["tau_max"])
        vals = (r["name"], tau_str, idx_str, r["index_bias"], mid_str, r["middle_bias"])
        print("  ".join(str(v).ljust(w) for v, w in zip(vals, widths)))

    print("\nCitations:")
    for fam, cite in CITATIONS.items():
        print(f"  {fam}: {cite}")


if __name__ == "__main__":
    main()
