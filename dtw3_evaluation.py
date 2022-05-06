
# TODO: If some kind of "graphs, images, etc" hyperparameter dependat is needed, make use of a wandb logger (it admit run summaries with numbers, vectors, images, etc; open mind, wandb is not only for trainings [If it do not works, maybe it will be needed to end the script with a log step to commit...])
# dtw3_validation wer contains all the code to apply dtw and return the accuracy... (UGLY!!! I will have to change it).
# dtw3_test test contains all the code to only generate predictions (OK!) but it needs 2 dataloaders instead of the only one it should.
# Remember torch has a Concat dataset made
# TODO: remake the list without random shuffling
# TODO: make submission files and memory
