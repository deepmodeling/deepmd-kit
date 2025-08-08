# unset PADDLE_ELASTIC_JOB_ID
# unset PADDLE_TRAINER_ENDPOINTS
# unset DISTRIBUTED_TRAINER_ENDPOINTS
# unset FLAGS_START_PORT
# unset PADDLE_ELASTIC_TIMEOUT
# export NNODES=1
# export PADDLE_TRAINERS_NUM=1
unset CUDA_DEVICE_MAX_CONNECTIONS

HDFS_USE_FILE_LOCKING=0 python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir "logs" dp --pd train input_torch.json -l dp_train.log

# NUM_WORKERS=0 HDFS_USE_FILE_LOCKING=0 python -m paddle.distributed.launch

# python -m paddle.distributed.launch \
#    --gpus=0,1,2,3 \
#    --ips=10.67.200.17,10.67.200.11,10.67.200.13,10.67.200.15 \
#    dp --pd train input_torch.json -l dp_train.log