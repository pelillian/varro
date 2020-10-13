LOG_FOLDER="logs/varro/algo/experiment/predict"
EXPERIMENT_FOLDER="simple_step_2020-Oct-05-20:51:37/"
mkdir $LOG_FOLDER
mkdir $LOG_FOLDER/$EXPERIMENT_FOLDER
for file in checkpoints/varro/algo/$EXPERIMENT_FOLDER*
do
    touch $LOG_FOLDER/$EXPERIMENT_FOLDER/$(basename $file)
    python -m varro.algo.experiment --model_type=fpga --problem_type=simple_step --popsize=100 --halloffamesize=0.2 --ngen=100000 --cxpb 0.0 --mutpb 0.5 --imutpb 0.4 --purpose=predict --input_data=01010101.npy --ckpt=checkpoints/varro/algo/$EXPERIMENT_FOLDER/$(basename $file) | grep "Loss" | cut -d " " -f 5 > $LOG_FOLDER/$EXPERIMENT_FOLDER/$(basename $file)
done
