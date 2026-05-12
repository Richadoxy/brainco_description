#!/usr/bin/env python3
"""Visualize self-collision points and forces for the Revo3 hand in MuJoCo.

Loads a pre-converted MJCF (.xml), sweeps each hinge joint through its range,
and enables MuJoCo's built-in contact-point visualization in the viewer.
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np


ARROW_LENGTH = 0.3
ARROW_WIDTH = 0.002
ARROW_RGBA = np.array([1.0, 0.2, 0.2, 0.95], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "xml",
        type=Path,
        help="Path to the MJCF .xml file.",
    )
    parser.add_argument(
        "--steps-per-joint",
        type=int,
        default=60,
        help="Animation steps used for each joint sweep.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Viewer/update rate during the sweep.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=12,
        help="Print the strongest contact every N frames.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of contacts written per frame to the CSV trace.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV file for the contact trace.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run the sweep headlessly and print contact data only.",
    )
    return parser.parse_args()


def geom_label(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    body_name = model.body(body_id).name or f"body_{body_id}"
    if body_name == "world":
        return "root_base(world)"
    return body_name


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        wrench_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_id, wrench_local)

        frame = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
        force_world = frame @ wrench_local[:3]
        magnitude = float(np.linalg.norm(force_world))
        if magnitude <= 0.0:
            continue

        records.append(
            {
                "contact_id": contact_id,
                "geom1": geom_label(model, contact.geom1),
                "geom2": geom_label(model, contact.geom2),
                "position": np.array(contact.pos, dtype=np.float64).copy(),
                "distance": float(contact.dist),
                "force_world": force_world,
                "force_norm": magnitude,
            }
        )

    records.sort(key=lambda item: float(item["force_norm"]), reverse=True)
    return records


def hinge_joint_ids(model: mujoco.MjModel) -> list[int]:
    return [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    ]


def joint_sweep_groups(model: mujoco.MjModel) -> list[list[int]]:
    hinge_ids = hinge_joint_ids(model)
    grouped_names = (
        ("thumb_cmp", ("thumb_CMP",)),
        ("thumb_cmr", ("thumb_CMR",)),
        ("thumb_mcp", ("thumb_MCP",)),
        ("thumb_pip", ("thumb_PIP",)),
        ("thumb_dip", ("thumb_DIP",)),
        ("fingers_mpr", ("index_MPR", "middle_MPR", "ring_MPR", "little_MPR")),
        ("fingers_mcp", ("index_MCP", "middle_MCP", "ring_MCP", "little_MCP")),
        ("fingers_pip", ("index_PIP", "middle_PIP", "ring_PIP", "little_PIP")),
        ("fingers_dip", ("index_DIP", "middle_DIP", "ring_DIP", "little_DIP")),
    )

    groups: list[list[int]] = []
    used: set[int] = set()
    for _, patterns in grouped_names:
        group = []
        for joint_id in hinge_ids:
            joint_name = model.joint(joint_id).name
            if any(pattern in joint_name for pattern in patterns):
                group.append(joint_id)
                used.add(joint_id)
        if group:
            groups.append(group)

    for joint_id in hinge_ids:
        if joint_id not in used:
            groups.append([joint_id])
    return groups


def neutral_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = np.zeros(model.nq, dtype=np.float64)
    for joint_id in hinge_joint_ids(model):
        qpos_addr = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        qpos[qpos_addr] = float(np.clip(0.0, lower, upper))
    return qpos


def sweep_values(start: float, lower: float, upper: float, steps: int) -> np.ndarray:
    steps = max(steps, 12)
    up_steps = max(steps // 2, 4)
    down_steps = max(steps - up_steps, 4)

    up = np.linspace(start, upper, up_steps, endpoint=False)
    down = np.linspace(upper, start, down_steps + 1, endpoint=True)[1:]
    return np.concatenate([up, down])


def configure_viewer(model: mujoco.MjModel, viewer) -> None:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = False
    model.vis.scale.contactwidth = 0.004
    model.vis.scale.contactheight = 0.03
    viewer.cam.lookat[:] = model.stat.center
    viewer.cam.distance = max(0.20, 1.8 * model.stat.extent)
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0


def update_contact_arrows(viewer, contacts: list[dict[str, object]]) -> None:
    if viewer is None or viewer.user_scn is None:
        return

    with viewer.lock():
        viewer.user_scn.ngeom = 0
        for contact in contacts:
            force_world = np.asarray(contact["force_world"], dtype=np.float64)
            force_norm = float(contact["force_norm"])
            if force_norm <= 0.0:
                continue
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                break

            start = np.asarray(contact["position"], dtype=np.float64)
            direction = force_world / force_norm
            end = start + ARROW_LENGTH * direction

            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                ARROW_RGBA,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                ARROW_WIDTH,
                start,
                end,
            )
            viewer.user_scn.ngeom += 1


def run_sweep(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    csv_writer: csv.DictWriter | None,
    viewer=None,
) -> None:
    base_qpos = neutral_qpos(model)
    groups = joint_sweep_groups(model)
    sleep_s = 0.0 if args.fps <= 0 else 1.0 / args.fps

    frame_index = 0
    for group in groups:
        joints = [model.joint(joint_id) for joint_id in group]
        joint_names = ", ".join(joint.name for joint in joints)
        qpos_addrs = [int(model.jnt_qposadr[joint_id]) for joint_id in group]
        ranges = [tuple(map(float, model.jnt_range[joint_id])) for joint_id in group]
        starts = [float(base_qpos[qpos_addr]) for qpos_addr in qpos_addrs]

        lower = min(range_[0] for range_ in ranges)
        upper = max(range_[1] for range_ in ranges)
        print(
            f"\nSweeping {joint_names}: qpos={qpos_addrs} from {lower:.4f} to {upper:.4f} rad"
        )

        for alpha in sweep_values(0.0, 0.0, 1.0, args.steps_per_joint):
            data.qpos[:] = base_qpos
            for qpos_addr, start, (joint_lower, joint_upper) in zip(qpos_addrs, starts, ranges):
                target = start + alpha * (joint_upper - start)
                data.qpos[qpos_addr] = np.clip(target, joint_lower, joint_upper)
            mujoco.mj_forward(model, data)

            contacts = contact_records(model, data)
            if frame_index % max(args.print_every, 1) == 0:
                if contacts:
                    strongest = contacts[0]
                    position = strongest["position"]
                    force_world = strongest["force_world"]
                    print(
                        "frame={:05d} joint={} q={:+.4f} ncon={} strongest={} <-> {} "
                        "point=({:+.4f}, {:+.4f}, {:+.4f}) "
                        "force=({:+.2f}, {:+.2f}, {:+.2f}) |F|={:.2f}".format(
                            frame_index,
                            joint_names,
                            alpha,
                            data.ncon,
                            strongest["geom1"],
                            strongest["geom2"],
                            position[0],
                            position[1],
                            position[2],
                            force_world[0],
                            force_world[1],
                            force_world[2],
                            strongest["force_norm"],
                        )
                    )
                else:
                    print(
                        f"frame={frame_index:05d} joint={joint_names} q={alpha:+.4f} ncon=0"
                    )

            if csv_writer is not None:
                rows = contacts[: max(args.top_k, 1)]
                if not rows:
                    csv_writer.writerow(
                        {
                            "frame": frame_index,
                            "joint": joint_names,
                            "qpos": alpha,
                            "contact_rank": -1,
                            "geom1": "",
                            "geom2": "",
                            "px": "",
                            "py": "",
                            "pz": "",
                            "fx": "",
                            "fy": "",
                            "fz": "",
                            "force_norm": 0.0,
                            "distance": "",
                            "ncon": data.ncon,
                        }
                    )
                for rank, row in enumerate(rows):
                    position = row["position"]
                    force_world = row["force_world"]
                    csv_writer.writerow(
                        {
                            "frame": frame_index,
                            "joint": joint_names,
                            "qpos": alpha,
                            "contact_rank": rank,
                            "geom1": row["geom1"],
                            "geom2": row["geom2"],
                            "px": position[0],
                            "py": position[1],
                            "pz": position[2],
                            "fx": force_world[0],
                            "fy": force_world[1],
                            "fz": force_world[2],
                            "force_norm": row["force_norm"],
                            "distance": row["distance"],
                            "ncon": data.ncon,
                        }
                    )

            if viewer is not None:
                update_contact_arrows(viewer, contacts)
                viewer.sync()
                time.sleep(sleep_s)

            frame_index += 1


def main() -> int:
    args = parse_args()
    xml_path = args.xml.expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    csv_file = None
    csv_writer = None
    if args.csv is not None:
        csv_path = args.csv.expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "frame",
                "joint",
                "qpos",
                "contact_rank",
                "geom1",
                "geom2",
                "px",
                "py",
                "pz",
                "fx",
                "fy",
                "fz",
                "force_norm",
                "distance",
                "ncon",
            ],
        )
        csv_writer.writeheader()

    print(f"Loaded {xml_path.name}")
    print(f"joints={model.njnt} hinge_joints={len(hinge_joint_ids(model))} geoms={model.ngeom}")
    print(
        "Viewer shows MuJoCo contact points and contact-force arrows. "
        "Terminal output reports the strongest contact force in world coordinates."
    )

    viewer_ctx = nullcontext()
    if not args.no_viewer:
        from mujoco import viewer as mujoco_viewer

        viewer_ctx = mujoco_viewer.launch_passive(
            model,
            data,
            show_left_ui=True,
            show_right_ui=True,
        )

    try:
        with viewer_ctx as viewer:
            if viewer is not None:
                configure_viewer(model, viewer)
            run_sweep(model, data, args, csv_writer, viewer)
    finally:
        if csv_file is not None:
            csv_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
