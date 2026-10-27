from __future__ import annotations

import torch


def compute_link_jacobians(pk_chain, q: torch.Tensor, frame_status: dict, frame_names):
    q = torch.atleast_2d(q)
    batch_size = q.shape[0]
    joint_names = pk_chain.get_joint_parameter_names()
    num_joints = len(joint_names)
    joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}

    frames = [pk_chain.find_frame(name) for name in pk_chain.get_joint_parent_frame_names()]
    jacobian_dict = {}

    transfer_transforms = {}
    for frame in frames:
        joint_idx = joint_name_to_idx[frame.joint.name]
        q_frame = q[:, joint_idx]
        if frame.joint.joint_type == "prismatic":
            q_frame = q_frame.unsqueeze(-1)
        transfer_transforms[joint_idx] = frame.get_transform(q_frame).get_matrix()

    for frame_name in frame_names:
        frame_transform = frame_status[frame_name].get_matrix()
        jacobian = torch.zeros(
            (batch_size, 6, num_joints),
            dtype=pk_chain.dtype,
            device=pk_chain.device,
        )

        rotation_world_frame = frame_transform[:, :3, :3]
        transform_joint_to_frame = torch.eye(
            4, dtype=pk_chain.dtype, device=pk_chain.device
        ).repeat(batch_size, 1, 1)

        frame_indices = pk_chain.parents_indices[pk_chain.frame_to_idx[frame_name]].tolist()
        for frame_idx in reversed(frame_indices):
            frame = pk_chain.find_frame(pk_chain.idx_to_frame[frame_idx])
            joint = frame.joint
            if joint.joint_type == "fixed":
                if joint.offset is not None:
                    transform_joint_to_frame = joint.offset.get_matrix() @ transform_joint_to_frame
                continue

            rotation_frame_joint = transform_joint_to_frame[:, :3, :3].transpose(1, 2)
            rotation_world_joint = rotation_world_frame @ rotation_frame_joint
            offset_joint_to_frame = transform_joint_to_frame[:, :3, 3][:, :, None]
            joint_axis = joint.axis[None, :, None].repeat(batch_size, 1, 1)

            if joint.joint_type in {"revolute", "continuous"}:
                linear_jacobian = rotation_world_joint @ torch.cross(joint_axis, offset_joint_to_frame, dim=1)
                angular_jacobian = rotation_world_joint @ joint_axis
            elif joint.joint_type == "prismatic":
                linear_jacobian = rotation_world_joint @ joint_axis
                angular_jacobian = torch.zeros(
                    (batch_size, 3, 1),
                    dtype=pk_chain.dtype,
                    device=pk_chain.device,
                )
            else:
                raise NotImplementedError(f"Unsupported joint type: {joint.joint_type}")

            joint_idx = joint_name_to_idx[joint.name]
            transform_joint_to_frame = transfer_transforms[joint_idx] @ transform_joint_to_frame
            jacobian[:, :, joint_idx] = torch.cat(
                [linear_jacobian[..., 0], angular_jacobian[..., 0]],
                dim=1,
            )

        jacobian_dict[frame_name] = jacobian

    return jacobian_dict
