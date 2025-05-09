TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=diffusiondrive_agent \
experiment_name=training_diffusiondrive_agent \
train_test_split=$TRAIN_TEST_SPLIT