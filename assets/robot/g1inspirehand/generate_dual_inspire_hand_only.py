from pathlib import Path
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "g1_upper_15dof_with_inspire_hand_FTP.urdf"
TARGET = HERE / "dual_inspire_hand_only.urdf"


def children_map(root):
    result = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        result.setdefault(parent, []).append((joint, child))
    return result


def collect_subtree(root, base_link_name):
    links_by_name = {link.attrib["name"]: link for link in root.findall("link")}
    child_joints = children_map(root)
    links = []
    joints = []
    stack = [base_link_name]
    seen = set()
    while stack:
        link_name = stack.pop()
        if link_name in seen:
            continue
        seen.add(link_name)
        links.append(links_by_name[link_name])
        for joint, child in child_joints.get(link_name, []):
            if "force_sensor" in child:
                continue
            joints.append(joint)
            stack.append(child)
    return links, joints


def add_virtual_base(robot, side):
    previous = "world"
    axes = [
        ("x", "prismatic", "1 0 0", "-1.0", "1.0"),
        ("y", "prismatic", "0 1 0", "-1.0", "1.0"),
        ("z", "prismatic", "0 0 1", "0.0", "1.5"),
        ("roll", "revolute", "1 0 0", "-3.1416", "3.1416"),
        ("pitch", "revolute", "0 1 0", "-3.1416", "3.1416"),
        ("yaw", "revolute", "0 0 1", "-3.1416", "3.1416"),
    ]
    for name, joint_type, axis, lower, upper in axes:
        link_name = f"{side}_base_{name}_link"
        ET.SubElement(robot, "link", {"name": link_name})
        joint = ET.SubElement(robot, "joint", {"name": f"{side}_base_{name}_joint", "type": joint_type})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(joint, "parent", {"link": previous})
        ET.SubElement(joint, "child", {"link": link_name})
        ET.SubElement(joint, "axis", {"xyz": axis})
        ET.SubElement(joint, "limit", {"lower": lower, "upper": upper, "effort": "1000", "velocity": "10"})
        previous = link_name

    joint = ET.SubElement(robot, "joint", {"name": f"{side}_base_mount_joint", "type": "fixed"})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": previous})
    ET.SubElement(joint, "child", {"link": f"{side}_base_link"})


def main():
    source_root = ET.parse(SOURCE).getroot()
    robot = ET.Element("robot", {"name": "dual_inspire_hand_only"})

    for material in source_root.findall("material"):
        robot.append(material)
    mujoco = source_root.find("mujoco")
    if mujoco is not None:
        robot.append(mujoco)

    ET.SubElement(robot, "link", {"name": "world"})

    for side in ("right", "left"):
        add_virtual_base(robot, side)
        links, joints = collect_subtree(source_root, f"{side}_base_link")
        for link in links:
            robot.append(link)
        for joint in joints:
            robot.append(joint)

    TARGET.write_text(ET.tostring(robot, encoding="unicode") + "\n")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
