export CUDA_VISIBLE_DEVICES=4,5,6,7
TRAIN_TEST_SPLIT=navtrain
CKPT_INIT=/fsave/tuzhirui/NAVSIM_v2.2/ckpt/diffusiondrive/diffusiondrive_init_1.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=diffusiondrive_agent \
experiment_name=training_diffusiondrive_agent \
train_test_split=$TRAIN_TEST_SPLIT \
trainer.params.max_epochs=10 \
agent.checkpoint_path=$CKPT_INIT \
cache_path="/fsave/tuzhirui/NAVSIM_v2.2/exp/training_cache_add_future_traj" \
use_cache_without_dataset=True  \
force_cache_computation=False \