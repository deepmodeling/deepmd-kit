# eval script
script_self=$(readlink -f "$0")
self_path=$(dirname "${script_self}")
DATA_PATH=$1
CKPT_PATH=$2
python -s ${self_path}/../eval.py --dataset_path=./$DATA_PATH --checkpoint_path=./$CKPT_PATH > log.txt 2>&1 &
