CONDA_ENV_NAME="meanaudio"
CONDA_HOME="/root/miniconda3"
CONDA_ENV_PATH="$CONDA_HOME/envs/$CONDA_ENV_NAME"
CONDA_COMMAND="$CONDA_HOME/bin/conda"


export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128


# 检查bash环境有没有init
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    echo "Conda init found"
    source $CONDA_HOME/etc/profile.d/conda.sh
fi

# 先检查是否存在该环境
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Conda environment $CONDA_ENV_NAME does not exist. Creating..."
    # 创建环境
    $CONDA_COMMAND create -n $CONDA_ENV_NAME python=3.12 -y
else
    echo "Conda environment $CONDA_ENV_NAME already exists"

    # 检查当前是否在目标环境中，如果是则先退出
    if [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV_NAME" ]; then
        echo "Currently in environment $CONDA_ENV_NAME, deactivating first..."
        conda deactivate
    fi

    # 删除环境
    $CONDA_COMMAND remove -n $CONDA_ENV_NAME --all -y
    # 创建环境
    $CONDA_COMMAND create -n $CONDA_ENV_NAME python=3.11 -y
fi

eval "$($CONDA_COMMAND shell.bash hook)"
conda activate $CONDA_ENV_NAME

cd /apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -e .

# install av-benchmark
cd av-benchmark
pip install -e .
cd ImageBind 
pip install -e .
cd ../MS-CLAP
pip install -e .
cd ../passt_hear21
pip install -e .

echo "Conda environment created successfully"