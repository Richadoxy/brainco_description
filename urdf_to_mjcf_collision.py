#!/usr/bin/env python3
"""Remove all <visual> elements from a URDF, keep <collision>, and convert to
MuJoCo MJCF (.xml). Only mass is kept (no inertia tensor)."""

import argparse
import shutil
import tempfile
import xml.etree.ElementTree as ET
import os
import mujoco


def process_urdf(tree: ET.ElementTree, urdf_dir: str, stage_dir: str):
    """Clean URDF: remove visual elements, remove inertial, stage meshes."""
    root = tree.getroot()

    for link in root.findall("link"):
        has_visual = link.find("visual") is not None
        for visual in link.findall("visual"):
            link.remove(visual)

        inertial = link.find("inertial")
        if inertial is not None:
            mass_elem = inertial.find("mass")
            mass_val = float(mass_elem.get("value", "0")) if mass_elem is not None else 0
            has_collision = link.find("collision") is not None
            if mass_val <= 0 and not has_visual and not has_collision:
                # Virtual link with no geometry: keep minimal inertial
                if mass_elem is not None:
                    mass_elem.set("value", "1e-06")
                inertia_elem = inertial.find("inertia")
                if inertia_elem is not None:
                    for attr in ("ixx", "iyy", "izz"):
                        if float(inertia_elem.get(attr, "0")) <= 0:
                            inertia_elem.set(attr, "1e-09")
            elif mass_val <= 0 or not has_collision:
                link.remove(inertial)

    mesh_abs_map = {}
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        abs_path = os.path.normpath(os.path.join(urdf_dir, filename))
        basename = os.path.basename(abs_path)
        mesh_abs_map[basename] = abs_path
        staged = os.path.join(stage_dir, basename)
        if not os.path.exists(staged):
            shutil.copy2(abs_path, staged)
        mesh.set("filename", basename)

    return mesh_abs_map


def urdf_to_mjcf(urdf_path: str, output_path: str):
    tree = ET.parse(urdf_path)
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    stage_dir = tempfile.mkdtemp(prefix="revo3_meshes_")

    mesh_abs_map = process_urdf(tree, urdf_dir, stage_dir)

    compiler = ET.SubElement(tree.getroot(), "compiler")
    compiler.set("meshdir", stage_dir)

    tmp_path = os.path.join(stage_dir, "tmp.urdf")
    try:
        ET.indent(tree, space="  ")
        tree.write(tmp_path, encoding="unicode", xml_declaration=True)

        model = mujoco.MjModel.from_xml_path(tmp_path)
        mujoco.mj_saveLastXML(output_path, model)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    # Post-process MJCF
    output_dir = os.path.dirname(os.path.abspath(output_path))

    # Build {mesh_basename: mass} from original URDF collision meshes
    mesh_mass_map = {}
    urdf_root = ET.parse(urdf_path).getroot()
    for link in urdf_root.findall("link"):
        for coll in link.findall("collision/geometry/mesh"):
            mesh_file = coll.get("filename", "")
            mesh_name = os.path.splitext(os.path.basename(mesh_file))[0]
            inertial = link.find("inertial/mass")
            mass = inertial.get("value", "0") if inertial is not None else "0"
            mesh_mass_map[mesh_name] = mass

    mjcf_tree = ET.parse(output_path)
    mjcf_root = mjcf_tree.getroot()

    # Fix mesh file paths
    for mesh in mjcf_root.findall(".//mesh"):
        basename = mesh.get("file", "")
        if basename in mesh_abs_map:
            abs_path = mesh_abs_map[basename]
            rel_path = os.path.relpath(abs_path, output_dir)
            mesh.set("file", rel_path)

    for geom in mjcf_root.iter("geom"):
        gmesh = geom.get("mesh", "")
        if gmesh in mesh_mass_map and float(mesh_mass_map[gmesh]) > 0:
            geom.set("mass", mesh_mass_map[gmesh])

    # Wrap top-level geoms and bodies in worldbody into a "base" body with free joint
    worldbody = mjcf_root.find("worldbody")
    if worldbody is not None:
        top_geoms = list(worldbody.findall("geom"))
        top_bodies = list(worldbody.findall("body"))
        if top_geoms or top_bodies:
            base_body = ET.Element("body", name="base", pos="0 0 0", euler="0 0 0")
            ET.SubElement(base_body, "joint", name="base_free", type="free")
            for g in top_geoms:
                worldbody.remove(g)
                base_body.append(g)
            for b in top_bodies:
                worldbody.remove(b)
                base_body.append(b)
            worldbody.insert(0, base_body)

    ET.indent(mjcf_tree, space="  ")
    mjcf_tree.write(output_path, encoding="unicode", xml_declaration=True)

    print(f"Saved MJCF to: {output_path}")
    print(f"  bodies={model.nbody}, joints={model.njnt}, geoms={model.ngeom}")


def main():
    parser = argparse.ArgumentParser(description="URDF -> MJCF converter (removes visual, keeps collision, mass-only)")
    parser.add_argument("urdf", help="Path to the input URDF file")
    parser.add_argument("-o", "--output", help="Output MJCF .xml path (default: same name with .xml)")
    args = parser.parse_args()

    output_path = args.output if args.output else os.path.splitext(args.urdf)[0] + ".xml"
    urdf_to_mjcf(args.urdf, output_path)


if __name__ == "__main__":
    main()
