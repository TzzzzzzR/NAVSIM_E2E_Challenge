TRAIN_TEST_SPLIT=navhard_two_stage

CKPT=/fsave/tuzhirui/NAVSIM_v2.2/ckpt/diffusiondrive/laneloss_1.ckpt

CACHE_PATH=/fsave/tuzhirui/NAVSIM_v2.2/exp/metric_cache_navhard_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=diffusiondrive_agent \
agent.config.latent=True \
worker=ray_distributed_no_torch \
agent.checkpoint_path=$CKPT \
experiment_name=diffusiondrive_agent_eval \
metric_cache_path=$CACHE_PATH \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \

