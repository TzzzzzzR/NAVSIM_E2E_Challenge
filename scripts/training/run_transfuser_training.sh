# TRAIN_TEST_SPLIT=navtrain

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
# agent=transfuser_agent \
# experiment_name=training_transfuser_agent \
# train_test_split=$TRAIN_TEST_SPLIT \

TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
        agent=transfuser_agent \
        experiment_name=training_transfuser_agent \
        train_test_split=$TRAIN_TEST_SPLIT \
        trainer.params.max_epochs=100 \
        cache_path="/fsave/tuzhirui/NAVSIM_v2.2/exp/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False 