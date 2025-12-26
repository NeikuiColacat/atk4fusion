export PYTHONHASHSEED=42

# 添加项目根目录到 PYTHONPATH
GPUS=1
NNODES=1
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29957}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/atk4fusion.py \
    --save_path "output/" \
    --gpus=$GPUS \
    --config=local_configs.NYUDepthv2.DFormer_Base \
    --continue_fpath=models/DFormer/checkpoints/trained/NYUv2_DFormer_Base.pth 
