########################################  Memory  ########################################
# clooect data
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


# 训练AE：
python -m memory.train_trajectory_ae train \
  --dataset datasets/wuji_grasp.npz \
  --output-dir runs/trajectory_ae \
  --epochs 1000 \
  --num-tokens 5 \
  --latent-dim 6 \
  --hidden-dim 128 \
  --batch-size 128 \
  --learning-rate 1e-4 \
  --rotation-matrix-loss-weight 1.0 \
  --virtual-frame-samples 0 \
  --traj_length 10 \
  --checkpoint-name ae_10.pt 


###################################### 训练点云AE：
python -m memory.train_pointcloud_ae train \
  --dataset datasets/inspire_grasp.npz \
  --output-dir runs/object_pointcloud_ae \
  --epochs 500 \
  --batch-size 16 \
  --latent-dim 128 \
  --hidden-dim 256 \
  --input-points 512 \
  --reconstruction-points 512 \
  --learning-rate 1e-3 \
  --device cuda
# 导出每条样本的点云 latent：
python -m memory.train_pointcloud_ae encode \
  --dataset datasets/inspire_grasp.npz \
  --checkpoint runs/object_pointcloud_ae/best.pt \
  --output runs/object_pointcloud_ae/inspire_pointcloud_latents.npz \
  --batch-size 128 \
  --device cuda
# 评估重建误差：
python -m memory.train_pointcloud_ae evaluate \
  --dataset datasets/inspire_grasp.npz \
  --checkpoint runs/object_pointcloud_ae/best.pt \
  --output runs/object_pointcloud_ae/evaluation.json \
  --batch-size 16 \
  --device cuda

#######################################
# 训练NFlow：
python -m memory.train_latent_flow train \
  --dataset datasets/wuji_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/ae_10.pt \
  --pointcloud-ae-checkpoint runs/object_pointcloud_ae/best.pt \
  --output-dir runs/latent_flow \
  --epochs 1000 \
  --traj_length 10 \
  --checkpoint-name flow_10.pt


# 真实轨迹重建回放测试AE：
# g1_upper_right_wuji
# fr3_inspire_tac
python -m testing.replay_trajectory_ae \
--dataset /home/cyrus/AgentGrasp/datasets/wuji_grasp.npz \
--checkpoint runs/trajectory_ae/best.pt \
--sample-index 3 \
--output runs/trajectory_ae/replay_0.npz \
task=grasp \
hand=g1_upper_right_wuji \
num_envs=1 \
headless=False \
task.env.armController=pose \
task.env.episodeLength=20 \
task.env.enablePointCloud=False



####################################  感知  ########################################

python -m perception.collect_object_partical_points \
  --num-envs 32 \
  --samples 17500 \
  --output datasets/objects \
  task=grasp \
  hand=g1_upper_right_wuji \
  task.env.armController=pose \
  task.env.enablePointCloud=True


python -m perception.test_rgbd_mask_frame \
  --dataset datasets/objects \
  --sample-index 100


python -m perception.train_rgbd_box_estimator \
  --dataset datasets/objects \
  --output runs/rgbd_box_estimator/pointr_best.pt \
  --epochs 500 \
  --batch-size 32 \
  --num-points 512 \
  --pretrained runs/rgbd_box_estimator/pointr_best.pt \
  --device cuda

python -m perception.test_rgbd_box_estimator \
  --dataset datasets/objects \
  --checkpoint runs/rgbd_box_estimator/pointr_canonical_best.pt \
  --sample-index 10
  --device cuda \
  --no-display



python /home/cyrus/AgentGrasp/datasets/objects/view_rgbd_pointcloud.py --shard 0 --sample 0

python /home/cyrus/AgentGrasp/datasets/objects/view_depth_reprojected_cloud.py --shard 0 --sample 0



##同时训练两个stage的PoinTr模型
python -u -m perception.train_rgbd_box_estimator \
  --dataset datasets/objects \
  --output runs/rgbd_box_estimator/pointr_two_stage_best.pt \
  --epochs 500 \
  --batch-size 2 \
  --num-points 512 \
  --pretrained external/PoinTr/pretrained/PCN_Pretrained.pth \
  --device cuda

python -m perception.test_rgbd_box_estimator \
  --dataset datasets/objects \
  --checkpoint runs/rgbd_box_estimator/pointr_two_stage_best.pt \
  --sample-index 100 \
  --device cuda \
  --no-display

########################################  wm  ########################################
python -m world_model.collect_latent_search_data \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --output datasets/latent_search_rgbd \
  --population-size 64 \
  --iterations 10 \
  task=grasp \
  hand=g1_upper_right_wuji \
  num_envs=64 \
  headless=True \
  task.env.armController=pose

python -m world_model.train_world_model \
  --latent-search-root datasets/latent_search_rgbd \
  --sample-glob 'iteration_*/candidate_*.npz' \
  --cache-dir datasets/latent_search_rgbd/.world_model_cache \
  --output-dir world_model/runs_world_model/g1_wuji_base_frame \
  --epochs 10 \
  --batch-size 8 \
  --lr 1e-5 \
  --device cuda

python -m world_model.test_world_model \
  +wm.dataset=datasets/latent_search_rgbd \
  +wm.checkpoint=world_model/runs_world_model/g1_wuji_base_frame/latest.pt \
  +wm.num_random_candidates=10 \
  +wm.random_seed=42 \
  +wm.enable_viewer=true \
  +wm.viewer_sleep_dt=0.1 \
  +wm.world_model_device=cuda \
  +wm.hold_viewer=true

########################################  test  ########################################
#群体搜索latent空间
# g1_upper_right_inspire
# fr3_inspire_tac
# g1_upper_right_wuji


python -m testing.search_latent_grasp \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --stage2-ae-checkpoint runs/trajectory_ae/ae_10.pt \
  --stage2-flow-checkpoint runs/latent_flow/flow_10.pt \
  --object-index 55 \
  --population-size 64 \
  --iterations 10 \
  --twostage_iterations 10 \
  --estimate_xyzrpy=True \
  task=grasp \
  hand=g1_upper_right_wuji \
  num_envs=64 \
  headless=True \
  task.env.armController=pose  \
  --action-noise-std 0.01 \
  --action-noise-fraction 0.25


python -m testing.test_g1_head_camera \
  task=grasp \
  hand=g1_upper_right_wuji \
  num_envs=1 \
  headless=False \
  task.env.armController=pose \
  task.env.enablePointCloud=False

python -m testing.test_latent_world_model \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --object-index 100 \
  --population-size 64 \
  --iterations 10 \
  +wm.checkpoint=/home/cyrus/DemoGrasp_wm/runs_world_model/2026-03-21_18-51-29/best.pt \
  task=grasp hand=g1_upper_right_inspire num_envs=1 headless=True \
  task.env.armController=pose


python -m testing.search_evaluator_guided_grasp \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/simulation_guided_grasp_pc_ae/ae_simulation_guided.pt \
  --flow-checkpoint runs/simulation_guided_grasp_pc_ae/flow_simulation_guided.pt \
  --evaluator-checkpoint runs/simulation_guided_grasp_pc_ae/evaluator.pt \
  --object-index 100 \
  --population-size 64 \
  --iterations 10 \
  task=grasp \
  hand=g1_upper_right_wuji \
  num_envs=1 \
  headless=False \
  task.env.armController=pose 


################################### train  ########################################
python -m memory.train_simulation_guided_grasp \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --population-size 64 \
  --object-count 175 \
  --iterations 200 \
  --action-noise-std 0.01 \
  --action-noise-fraction 0.25 \
  task=grasp \
  hand=g1_upper_right_wuji \
  num_envs=175 \
  headless=True \
  task.env.armController=pose 