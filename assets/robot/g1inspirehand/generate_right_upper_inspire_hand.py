"""Generate the fixed-waist, right-arm G1 + Inspire hand asset used by grasp."""

from pathlib import Path
import copy
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "g1_upper_15dof_with_inspire_hand_FTP.urdf"
TARGET = HERE / "g1_upper_right_inspire_hand.urdf"


def _children(root):
    result = {}
    for joint in root.findall("joint"):
        result.setdefault(joint.find("parent").get("link"), []).append(joint)
    return result


def _right_upper_subtree(root):
    links = {link.get("name"): link for link in root.findall("link")}
    children = _children(root)
    # Keep the pelvis, fixed waist, torso, and the complete right arm/hand.
    stack = ["pelvis"]
    selected_links = set()
    selected_joints = []
    while stack:
        parent = stack.pop()
        if parent in selected_links:
            continue
        selected_links.add(parent)
        for joint in children.get(parent, []):
            child = joint.find("child").get("link")
            if child.startswith("left_") or child in {
                "head_link",
                "logo_link",
                "d435_link",
                "mid360_link",
                "imu_in_torso",
                "imu_in_pelvis",
            }:
                continue
            selected_joints.append(joint)
            stack.append(child)
    return [link for link in root.findall("link") if link.get("name") in selected_links], selected_joints


def main():
    source_root = ET.parse(SOURCE).getroot()
    robot = ET.Element("robot", {"name": "g1_upper_right_inspire_hand"})
    for material in source_root.findall("material"):
        robot.append(copy.deepcopy(material))
    mujoco = source_root.find("mujoco")
    if mujoco is not None:
        robot.append(copy.deepcopy(mujoco))

    links, joints = _right_upper_subtree(source_root)
    for link in links:
        robot.append(copy.deepcopy(link))
    for joint in joints:
        joint_copy = copy.deepcopy(joint)
        # Grasp uses a fixed robot base.  Lock G1's three waist joints so the
        # remaining controllable DOFs are 7 arm + 12 Inspire hand joints.
        if joint_copy.get("name") in {
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        }:
            joint_copy.set("type", "fixed")
            for child_name in ("axis", "limit", "dynamics"):
                node = joint_copy.find(child_name)
                if node is not None:
                    joint_copy.remove(node)
        robot.append(joint_copy)

    ET.indent(robot, space="  ")
    TARGET.write_text(ET.tostring(robot, encoding="unicode") + "\n")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
