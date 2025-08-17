#!/bin/bash

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

export HF_ENDPOINT=https://hf-mirror.com


# 参数解析
MASTER_ADDR=$1
MASTER_PORT=$2
WORLD_SIZE=$3
GPUS_PER_NODE=$4
shift 4
PYTHON_ARGS="$@"

# # 计算本节点的 node_rank (INDEX) - torchrun 需要这个变量
# LOCAL_IP=$(hostname -I | awk '{print $1}')
# # 从 MASTER_ADDR 和其他环境变量推断节点列表，或者使用预设的节点列表
# # 这里我们基于 IP 地址计算 INDEX
# if [[ "$LOCAL_IP" == "$MASTER_ADDR" ]]; then
#     INDEX=0  # 主节点总是 0
# else
#     # 简单的基于 IP 最后一段的计算，可能需要根据实际情况调整
#     LAST_OCTET=$(echo $LOCAL_IP | cut -d. -f4)
#     MASTER_LAST_OCTET=$(echo $MASTER_ADDR | cut -d. -f4)
#     INDEX=$(( ($LAST_OCTET - $MASTER_LAST_OCTET) % 256 ))
#     if [ $INDEX -lt 0 ]; then
#         INDEX=$(( $INDEX + 256 ))
#     fi
# fi

# 设置环境变量（与原始脚本完全一致）
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export GPUS_PER_NODE=$GPUS_PER_NODE
# NODE_RANK不再需要，因为日志配置已改用RANK环境变量
# export NODE_RANK=$INDEX

# 设置NCCL参数（与原始脚本完全一致）
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=false
export TERM=xterm-256color
# 设置时区为北京时间
export TZ=Asia/Shanghai

# 激活环境（与原始脚本一致）
CONDA_ENV_NAME="meanaudio"
CONDA_HOME="/root/miniconda3"
CONDA_COMMAND="$CONDA_HOME/bin/conda"
eval "$($CONDA_COMMAND shell.bash hook)"
conda activate $CONDA_ENV_NAME


echo "conda activate $CONDA_ENV_NAME"
echo "which python: $(which python)"; python -V; echo "which torchrun: $(which torchrun)"


# SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd /apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$((WORLD_SIZE / GPUS_PER_NODE)) \
    --node_rank=$INDEX \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py $PYTHON_ARGS
