export CUDA_VISIBLE_DEVICES=4,5,6,7
TRAIN_TEST_SPLIT=navtrian

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=diffusiondrive_agent \
experiment_name=training_diffusiondrive_agent \
train_test_split=$TRAIN_TEST_SPLIT \
trainer.params.max_epochs=50 \
agent.checkpoint_path=$CKPT_INIT \
cache_path="/fsave/tuzhirui/DiffusionDrive/exp/training_cache_in_diffusiondrive/" \
use_cache_without_dataset=True  \
force_cache_computation=False \