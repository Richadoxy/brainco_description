#!/usr/bin/env python3
"""Load a MuJoCo MJCF XML and print each joint's body inertia computed from mesh geometry.

MuJoCo automatically computes inertia from mesh volume when no explicit inertial
is provided. This script extracts those values in a format ready to paste back
into URDF <inertial> blocks.

Usage:
    python compute_inertia.py revo3_system/urdf/revo3_right.xml
"""

import argparse
import math
import numpy as np
import mujoco


def quat_to_rpy(quat_wxyz: np.ndarray) -> tuple:
    """MuJoCo quaternion (w,x,y,z) -> roll, pitch, yaw (radians)."""
    w, x, y, z = quat_wxyz
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", help="Path to the MJCF .xml file")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)

    print(f"{'Joint':<35} {'Body':<30} {'Mass':>10} {'Ixx':>14} {'Iyy':>14} {'Izz':>14}")
    print("-" * 117)

    for jid in range(model.njnt):
        jnt = model.joint(jid)
        body_id = int(model.jnt_bodyid[jid])
        body = model.body(body_id)

        mass = float(body.mass[0])
        ixx, iyy, izz = body.inertia
        ipos = body.ipos
        rpy = quat_to_rpy(body.iquat)

        print(f"{jnt.name:<35} {body.name:<30} {mass:>10.6f} {ixx:>14.10f} {iyy:>14.10f} {izz:>14.10f}")

    print()
    print("URDF <inertial> template for each body (copy-paste into URDF):")
    print()

    for jid in range(model.njnt):
        body_id = int(model.jnt_bodyid[jid])
        body = model.body(body_id)
        name = body.name

        mass = float(body.mass[0])
        ixx, iyy, izz = body.inertia
        ipos = body.ipos
        rpy = quat_to_rpy(body.iquat)

        print(f'  <!-- {name} -->')
        print(f'  <inertial>')
        print(f'    <origin rpy="{rpy[0]:.10f} {rpy[1]:.10f} {rpy[2]:.10f}" xyz="{ipos[0]:.10f} {ipos[1]:.10f} {ipos[2]:.10f}" />')
        print(f'    <mass value="{mass:.10f}" />')
        print(f'    <inertia ixx="{ixx:.10f}" ixy="0" ixz="0" iyy="{iyy:.10f}" iyz="0" izz="{izz:.10f}" />')
        print(f'  </inertial>')
        print()


if __name__ == "__main__":
    main()
