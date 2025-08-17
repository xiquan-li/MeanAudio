#!/bin/bash
export PDSH_RCMD_TYPE=ssh
export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
echo $node_ip
pdsh -w $node_ip "cd /apdcephfs_gy4/share_302507476/xiquanli/TTA/MeanAudio; bash scripts/create_env.sh"
