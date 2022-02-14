stage=${1:-'finetune'}      # `finetune` or `pretrain`
device=${2:-'0'}            # GPU device ID
method_name=${3:-'gmpt_cl'} # can be `gmpt_cl`, `gmpt_sup` or `gmpt_suppp`
pretrain_epoch=${4:-'5'}    # suggested `5` for `gmpt_cl` and `100` for `gmpt_sup` and `gmpt_suppp`
gnn_type=${5:-'gin'}        # `gin`, 'gcn', 'graphsage' or `gat`

num_workers=4
model_path=bio_pretrain_model/${method_name}_${gnn_type}
dataset_path=dataset/Bio
ft_lr=0.0001

if [ ${stage} == 'finetune' ]
then
    # Fine-tuning
    for runseed in {0..9}
    do
        python bio_finetune.py \
            --model_file ${model_path}.pth.${pretrain_epoch} \
            --filename ${method_name}_${gnn_type}_${pretrain_epoch} \
            --device ${device} \
            --runseed ${runseed} \
            --gnn_type ${gnn_type} \
            --num_workers ${num_workers} \
            --dataset_path ${dataset_path} \
            --lr ${ft_lr}
    done
else
    # Pre-training
    python pretrain_${method_name}.py \
        --model_file ${model_path} \
        --device ${device} \
        --num_workers ${num_workers} \
        --dataset_path ${dataset_path} \
        --epochs ${pretrain_epoch} \
        --gnn_type ${gnn_type}
fi
