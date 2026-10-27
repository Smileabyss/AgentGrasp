# AgentGrasp

AgentGrasp is organized around callable grasping modules. It retains the AE/normalizing-flow trajectory memory, the point-cloud world model, Isaac Gym playback/search, and RGB-D collection. The copied project supports only these robot configurations: `g1_upper_right_inspire`, `fr3_inspire_tac`, and `g1_upper_right_wuji`.

## Project modules

- `memory/`: grasp dataset collection, trajectory AE/VQ-VAE, latent normalizing flow, joint training, encode/sample, symmetry analysis, and trajectory visualization.
- `world_model/`: point-cloud trajectory dataset storage, world-model training, rollout evaluation, PSO planner, and the model checkpoint. `world_model/samples/` contains the copied training episodes; preprocessing cache is generated on demand.
- `testing/`: Isaac Gym trajectory replay, CEM search, world-model-only latent search, and head-camera simulation.
- `perception/`: RGB-D collection and partial-to-full point-cloud Flow Matching.
- `agent/`: Codex-assisted refinement of a failed latent-search result, followed by simulator verification.
- `tasks/`, `assets/`, `algo/`: shared Isaac Gym environment, the three robot asset sets, object assets, and inference-only policy encoders used by collection.

Training, testing, collection, and visualization entry points live in their feature modules. Run them from this directory with `python -m <module>.<script>`.

## Memory

Collect trajectories for AE/NF training (the pretrained one-step checkpoint is used only to produce successful samples):

```bash
python -m memory.collect_grasp_dataset \
  --num-episodes 1 \
  --output datasets/inspire_grasp_test.npz \
  task=grasp train=PPOOneStep test=True num_envs=175 \
  task.env.observationType="eefpose+objinitpose+objpcl" \
  task.env.armController=pose \
  task.env.asset.multiObjectList="union_ycb_unidex/union_ycb_debugset.yaml" \
  task.env.randomizeTrackingReference=True \
  task.env.randomizeGraspPose=True \
  task.env.trackingReferenceFile=tasks/grasp_ref_inspire.pkl \
  task.env.trackingReferenceLiftTimestep=13 \
  task.env.episodeLength=50 \
  task.env.enablePointCloud=True \
  train.params.is_vision=True \
  checkpoint="ckpt/inspire.pt" \
  --no-symmetry-augmentation
```

Train AE and NF, then search or replay decoded actions:

```bash
python -m memory.train_trajectory_ae train --dataset datasets/inspire_grasp.npz --output-dir runs/trajectory_ae --epochs 1000 --num-tokens 5 --latent-dim 6 --hidden-dim 128 --batch-size 128
python -m memory.train_latent_flow train --dataset datasets/inspire_grasp.npz --ae-checkpoint runs/trajectory_ae/best.pt --output-dir runs/latent_flow --epochs 500
python -m testing.replay_latent_flow --dataset datasets/inspire_grasp.npz --ae-checkpoint runs/trajectory_ae/best.pt --flow-checkpoint runs/latent_flow/best.pt --object-index 50 task=grasp hand=fr3_inspire_tac num_envs=1 headless=False task.env.armController=pose
```

Use `--traj_length N` to train a fixed-length model from stride-one windows of
the original 20-step trajectories. For example, the following commands train
independent 10-step AE and flow checkpoints; omit the option (or set it to 20)
to keep the full-trajectory behavior. Use a different output directory for
each length.

```bash
python -m memory.train_trajectory_ae train --dataset datasets/wuji_grasp.npz --output-dir runs/trajectory_ae_10 --checkpoint-name ae_10.pt --traj_length 10
python -m memory.train_latent_flow train --dataset datasets/wuji_grasp.npz --ae-checkpoint runs/trajectory_ae_10/ae_10.pt --output-dir runs/latent_flow_10 --checkpoint-name flow_10.pt --traj_length 10
```

## World model

World-model training reads episodes from `world_model/samples/` and writes checkpoints under `world_model/runs_world_model/`:

```bash
python -m world_model.train_world_model --samples-root world_model/samples --output-dir world_model/runs_world_model --epochs 50 --batch-size 8
python -m world_model.test_world_model +wm.checkpoint=world_model/runs_world_model/2026-03-21_18-51-29/best.pt task=grasp hand=g1_upper_right_inspire num_envs=1 headless=True task.env.armController=pose
```

`testing.test_latent_world_model` uses Isaac Gym to read the object pose/cloud once, then scores every latent candidate in the point-cloud world model. It does not use Isaac Gym for candidate fitness. The best candidate is replayed in Isaac Gym after search.

## Isaac Gym search and Codex refinement

```bash
python -m testing.search_latent_grasp \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --stage2-ae-checkpoint runs/trajectory_ae_10/ae_10.pt \
  --stage2-flow-checkpoint runs/latent_flow_10/flow_10.pt \
  --object-index 120 --population-size 64 --iterations 10 \
  --twostage_iterations 10 \
  task=grasp hand=g1_upper_right_wuji num_envs=64 headless=True \
  task.env.armController=pose

python -m agent.agent_search_latent_grasp \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --object-index 10 --population-size 64 --iterations 10 \
  --agent think=True \
  task=grasp hand=g1_upper_right_wuji num_envs=64 headless=True \
  task.env.armController=pose
```

The Codex option requires a working `codex login` session. The agent receives point clouds, robot state/limits and the selected URDF/IK tool; its proposed trajectory is checked in Isaac Gym.

## RGB-D and camera output

The head-camera test runs the configured camera in Isaac Gym and can record output under `--record-path`. The RGB-D collector writes sampled RGB/depth observations, object masks, complete point clouds, and pose labels as shards; the estimator CLI back-projects masked depth into partial object point clouds and trains/evaluates the two-stage PoinTr models from those shards.

```bash
python -m testing.test_g1_head_camera --record-path runs/head_camera_test task=grasp hand=g1_upper_right_inspire num_envs=1 headless=False task.env.armController=pose task.env.enablePointCloud=False
python -m perception.collect_rgbd_object_data --num-envs 32 --samples 10000 --output datasets/rgbd_object task=grasp hand=g1_upper_right_wuji task.env.armController=pose task.env.enablePointCloud=True
python -m perception.collect_object_partical_points --num-envs 32 --samples 17500 --output datasets/object_partial_points task=grasp hand=g1_upper_right_wuji task.env.armController=pose task.env.enablePointCloud=True
python datasets/rgbd_object/view_rgbd_pointcloud.py --data-dir datasets/object_partial_points --shard 0 --sample 0
python -m perception.train_rgbd_box_estimator --dataset datasets/objects --output runs/rgbd_box_estimator/pointr_two_stage_best.pt --epochs 500 --batch-size 2 --device cuda
python -m perception.test_rgbd_box_estimator --dataset datasets/objects --checkpoint runs/rgbd_box_estimator/pointr_two_stage_best.pt --sample-index 0
```

`train_rgbd_box_estimator` jointly trains two PoinTr models: stage 1 completes the partial RGB-D cloud in its estimated frame, and stage 2 maps that completion to the object's asset-root frame. The test command reports each stage's Chamfer L1, displays their point clouds, and estimates the estimated-to-object-frame transform by ICP so its rotation and translation can be compared with the saved pose label. ICP's pose estimate can be ambiguous for symmetric objects. Two PoinTr models use more GPU memory than one, so reduce `--batch-size` if needed.

The balanced object RGB-D collector uses the 175-object `union_ycb_debugset.yaml` by default and creates 100 samples per object (17,500 total). It loads each object into the G1 Wuji grasp scene, renders RGB, depth, and segmentation from the head camera, and saves the complete 512-point asset cloud in camera coordinates with the object pose. Every object receives the same number of accepted samples; `--samples` must be divisible by the configured object count. Each shard follows the `rgbd_object` schema (`rgb`, `depth`, `bbox`, `object_mask`, `object_asset_id`, `points_camera`, and `pose_camera`).

`world_model.collect_latent_search_data` randomly selects one object row from the grasp dataset (or accepts `--object-index`) and records every candidate evaluated during each CEM iteration. Each `iteration_NNN/candidate_NNNN.npz` contains post-action RGB/depth, object and hand segmentation masks, a depth-derived visible object cloud, a per-frame Wuji URDF hand-mesh cloud transformed by the hand-link poses, the complete object-local asset cloud, robot joint position/velocity, and the candidate latent/action trajectory. The selected object's dataset pose is held fixed for all candidates and iterations. Camera images default to 320x240 to keep collection size manageable. The hand mask records visible pixels only and can be empty when the hand is outside the camera view; the hand mesh cloud still contains the complete modeled hand.

```bash
python -m world_model.collect_latent_search_data \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --population-size 64 \
  --iterations 5 \
  --output datasets/latent_search_rgbd \
  task=grasp hand=g1_upper_right_wuji num_envs=64 headless=True \
  task.env.armController=pose
```

The source project provides Isaac Gym simulation and camera capture. It does not include a physical robot driver or real-camera acquisition adapter.
