TRAIN_FILE=$1
EXPERIMENT_NAME=$2

openai tools fine_tunes.prepare_data -f $TRAIN_FILE

openai api fine_tunes.create \
    --training_file $TRAIN_FILE \
    --model davinci \
    --suffix $EXPERIMENT_NAME \
    --n_epochs 2 \
    --prompt_loss_weight 0