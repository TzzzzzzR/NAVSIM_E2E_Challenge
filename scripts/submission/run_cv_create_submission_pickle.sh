TEAM_NAME="https://github.com/TzzzzzzR/NAVSIM_E2E_Challenge"
AUTHORS="ZhiruiTu"
EMAIL="jerrytu0656@gmail.com"
INSTITUTION="South_China_University_Of_Technology"
COUNTRY="China"

TRAIN_TEST_SPLIT=private_test_hard_two_stage
SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/sensor_blobs
SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/private_test_hard_two_stage/openscene_meta_datas

CHECKPOINT=/fsave/tuzhirui/NAVSIM_v2.2/ckpt/diffusiondrive/laneloss_1.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=diffusiondrive_agent \
agent.config.latent=True \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=submission_diffusiondrive \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
