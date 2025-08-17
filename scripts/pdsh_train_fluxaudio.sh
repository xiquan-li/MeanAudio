#!/bin/bash

function find_free_port() {
    # 端口搜索循环（与原始脚本保持一致）
    local start=23456
    local end=33456
    local free_port=23456
    for port in $(seq $start 100 $end)
    do
        # 使用lsof命令检查端口是否被占用，如果未被占用，那么将此端口号赋值给变量并退出搜索
        (echo >/dev/tcp/localhost/$port) >/dev/null 2>&1
        if [[ $? -eq 1 ]]; then
            free_port=$port
            break
        fi
    done
    echo $free_port
}

function get_host_list() {
    # 从hostfile中读取主机列表
    local hostfile="/etc/taiji/hostfile"
    if [ -f "$hostfile" ]; then
        # 提取IP地址，去掉slots信息
        cat "$hostfile" | grep -v "^#" | awk '{print $1}' | sort | uniq
    else
        echo "localhost"
    fi
}

function get_total_gpus() {
    # 计算所有机器的GPU总数
    local hosts=($(get_host_list))
    local total_gpus=0
    local gpus_per_node=${GPUS_PER_NODE:-8}
    
    for host in "${hosts[@]}"; do
        if [ "$host" = "localhost" ] || [ "$host" = "$(hostname -I | awk '{print $1}')" ]; then
            # 本机直接检测
            local gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
            total_gpus=$((total_gpus + gpu_count))
        else
            # 远程机器假设有GPUS_PER_NODE个GPU
            total_gpus=$((total_gpus + gpus_per_node))
        fi
    done
    
    echo $total_gpus
}

# 环境设置
CONDA_ENV_NAME=meanaudio
CONDA_HOME="/root/miniconda3"
CONDA_ENV_PATH="$CONDA_HOME/envs/$CONDA_ENV_NAME"
CONDA_COMMAND="$CONDA_HOME/bin/conda"
eval "$($CONDA_COMMAND shell.bash hook)"
conda activate $CONDA_ENV_NAME

# 多机多卡参数
HOSTFILE="/etc/taiji/hostfile"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}  # 每个节点的GPU数

# 获取本机IP（与原始脚本保持一致）
if [[ -z "$LOCAL_IP" ]]; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    if [[ -z "$LOCAL_IP" ]]; then
        LOCAL_IP=$(ip route get 8.8.8.8 | awk 'NR==1 {print $7}')
    fi
fi
echo "当前机器IP: ${LOCAL_IP}"


echo "使用PyTorch DDP多机多卡训练..."

# 获取主机列表
HOSTS=($(get_host_list))
NUM_NODES=${#HOSTS[@]}
TOTAL_GPUS=$(get_total_gpus)

# SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT_DIR='/apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio'

# 设置主节点（使用当前机器的LOCAL_IP作为主节点，与原始脚本保持一致）
export MASTER_ADDR=${LOCAL_IP}
export MASTER_PORT=$(find_free_port)
export GPUS_PER_NODE=${GPUS_PER_NODE}

echo "多机多卡配置:"
echo "  节点数: ${NUM_NODES}"
echo "  每节点GPU数: ${GPUS_PER_NODE}"
echo "  总GPU数: ${TOTAL_GPUS}"
echo "  主节点: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  节点列表: ${HOSTS[*]}"
echo "  当前节点IP: ${LOCAL_IP}"

# 设置NCCL参数
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
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=false
export TERM=xterm-256color
# 设置时区为北京时间
export TZ=Asia/Shanghai

export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128

export PDSH_RCMD_TYPE=ssh
export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
echo $node_ip


#### Model config
# MODEL='fluxaudio_m_full_30'
MODEL='fluxaudio_s_full'
# MODEL='fluxaudio_fm'
TEXT_ENCODER_NAME='t5_clap'
TEXT_C_DIM=512


#### Data config
PER_GPU_BATCH_SIZE=64
BATCH_SIZE=$((TOTAL_GPUS * PER_GPU_BATCH_SIZE))
EVAL_BATCH_SIZE=4
NUM_WORKERS=10

#### Training config
NUM_ITERATIONS=50_000
MINI_TRAIN=False
USE_MEANFLOW=False   # for runner config 
USE_WANDB=True
# WEIGHTS='/apdcephfs_gy4/share_302507476/xiquanli/exps/FluxAudio/AWM_8kh_10s_bsz4096_numgpus64_niter_T5_CLAP_flowmatching/AWM_8kh_10s_bsz4096_numgpus64_niter_T5_CLAP_flowmatching_ema_final.pth'

# EXP_ID=FMA_Jamendo_30s_bsz${BATCH_SIZE}_numgpus${TOTAL_GPUS}_niter${NUM_ITERATIONS}_T5_CLAP_${MODEL}
EXP_ID=AWM_8kh_10s_bsz${BATCH_SIZE}_numgpus${TOTAL_GPUS}_niter${NUM_ITERATIONS}_T5_CLAP_${MODEL}

if [ "$MINI_TRAIN" = "True" ]; then
    echo "======================== Mini train ========================="
    EXP_ID=debug_mf
    VAL_INTERVAL=100
    EVAL_INTERVAL=100
    SAVE_EVAL_INTERVAL=100
    SAVE_WEIGHTS_INTERVAL=100
    SAVE_CHECKPOINT_INTERVAL=100
    EMA_CHECKPOINT_EVERY=50
    NUM_ITERATIONS=200
    BATCH_SIZE=$((TOTAL_GPUS * 4))
    USE_WANDB=False
else
    VAL_INTERVAL=10_000
    EVAL_INTERVAL=10_000
    SAVE_EVAL_INTERVAL=10_000
    SAVE_WEIGHTS_INTERVAL=10_000
    SAVE_CHECKPOINT_INTERVAL=10_000
    EMA_CHECKPOINT_EVERY=10_000
fi

#### Run training
pdsh -w $node_ip "cd ${SCRIPT_DIR} && bash ${SCRIPT_DIR}/scripts/train_multinodes.sh ${MASTER_ADDR} ${MASTER_PORT} ${TOTAL_GPUS} ${GPUS_PER_NODE} \
        --config-name=train_config.yaml \
        exp_id=${EXP_ID} \
        model=${MODEL} \
        batch_size=${BATCH_SIZE} \
        eval_batch_size=${EVAL_BATCH_SIZE} \
        num_iterations=${NUM_ITERATIONS} \
        num_workers=${NUM_WORKERS} \
        text_encoder_name=${TEXT_ENCODER_NAME} \
        data_dim.text_c_dim=${TEXT_C_DIM} \
        mini_train=${MINI_TRAIN} \
        val_interval=${VAL_INTERVAL} \
        eval_interval=${EVAL_INTERVAL} \
        save_eval_interval=${SAVE_EVAL_INTERVAL} \
        save_weights_interval=${SAVE_WEIGHTS_INTERVAL} \
        save_checkpoint_interval=${SAVE_CHECKPOINT_INTERVAL} \
        ema.checkpoint_every=${EMA_CHECKPOINT_EVERY} \
        pin_memory=False \
        cfg_strength=4.5 \
        use_meanflow=${USE_MEANFLOW} \
        ++use_rope=True \
        ++use_wandb=${USE_WANDB} \
        ++do_eval=True"