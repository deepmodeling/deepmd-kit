unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export NNODES=1
export PADDLE_TRAINERS_NUM=1
unset CUDA_DEVICE_MAX_CONNECTIONS

# export GLOG_v=7

export FLAGS_check_cuda_error=1
export FLAGS_call_stack_level=3


# export FLAGS_cudnn_deterministic=True
# export FLAGS_embedding_deterministic=1 

export PYTHONPATH=/root/paddlejob/workspace/env_run/xuexixi/Paddle/build/:$PYTHONPATH
export PYTHONPATH=/root/paddlejob/workspace/env_run/xuexixi/Paddle/build/python/:$PYTHONPATH

rm -rf core*
rm -rf logs
# ps -ef|grep dpa3|awk '{print $2}'|xargs kill -9
source /root/paddlejob/workspace/env_run/xuexixi/pybot/bin/activate

# nsys_args="/opt/nvidia/nsight-systems/2023.2.1/bin/nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true -o ./tmp"
${nsys_args} python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" --log_dir "logs" dp --pd train input_torch.json -l dp_train.log
