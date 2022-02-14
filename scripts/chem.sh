stage=${1:-'finetune'}      # `finetune` or `pretrain`
device=${2:-'0'}            # GPU device ID
ft_lr=${3:-'0.001'}         # fine-tuning learning rate, suggested 0.01 for toxcast
method_name=${4:-'gmpt_cl'} # can be `gmpt_cl` or `gmpt_suppp`
pretrain_epoch=${5:-'10'}    # suggested `10` for `gmpt_cl` and `100` for `gmpt_suppp`

num_workers=4
mode=chem
model_path=chem_pretrain_model/${method_name}
dataset_path=dataset/Chem
batch_size=32

if [ ${stage} == 'finetune' ]
then
    # Fine-tuning
    for runseed in {0..9}
    do
        for dataset in sider clintox bace bbbp tox21 toxcast hiv muv
        do
            python chem_finetune.py \
                --input_model_file ${model_path}.pth.${pretrain_epoch} \
                --filename ${method_name}_${dataset}_${pretrain_epoch} \
                --device ${device} \
                --runseed ${runseed} \
                --num_workers ${num_workers} \
                --dataset_path ${dataset_path} \
                --dataset ${dataset} \
                --lr ${ft_lr}
        done
    done
else
    # Pre-training
    python pretrain_${method_name}.py \
        --mode ${mode} \
        --model_file ${model_path} \
        --device ${device} \
        --num_workers ${num_workers} \
        --dataset_path ${dataset_path} \
        --epochs ${pretrain_epoch} \
        --batch_size ${batch_size}
fi
