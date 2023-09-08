exp_config=neurips2021-celeba.yml
STYLEGAN_PKL=pretrained/stylegan/neurips2021-celeba-stylegan/network-snapshot-002298.pkl
exp_id=CelebA.1.data
lr=1e-4 
lambda_kl=1e-3
l=0-9
method=layeredflow
permute=shuffle
K=10
flow_coupling=additive
L=3
flow_use_actnorm=1
glow=1
lambda_miner_entropy=0
lambda_attack=1
run_target_feat_eval=1

for fixed_id in {0..99}; do
output_dir=results/celeba-id${fixed_id}

cmd="attack_stylegan.py \
--flow_permutation ${permute} \
--flow_K ${K} \
--flow_glow ${glow} \
--flow_coupling ${flow_coupling} \
--flow_L ${L} \
--flow_use_actnorm ${flow_use_actnorm} \
--network ${STYLEGAN_PKL} \
--l_identity ${l} \
--db 0 \
--run_target_feat_eval ${run_target_feat_eval} \
--method ${method} \
--exp_config ${exp_config} \
--prior_model 0 \
--fixed_id ${fixed_id} \
--save_model_epochs 10,20,30 \
--resume $1 \
--patience 1000 \
--output_dir ${output_dir} \
--log_iter_every 100  \
--viz_every 1 \
--epochs 31 \
--save_samples_every 1000 \
--lambda_weight_reg 0 \
--lambda_attack 1 \
--lambda_prior 0 \
--lambda_miner_entropy ${lambda_miner_entropy} \
--lambda_kl ${lambda_kl} \
--ewc_type None \
--lr ${lr} \
"

if [ $1 == 0 ] 
then
python $cmd
break 100
else
# Run sbatch, and create a file with name based on job-id
RES=$(sbatch <<< \
"#!/bin/bash
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -p rtx6000
#SBATCH --time=200:00:00
#SBATCH --output=logs/attack-stylegan-%j-out.txt
#SBATCH --error=logs/attack-stylegan-%j-err.txt
#SBATCH --qos=normal


echo $cmd
python $cmd
") && touch jobs/job${RES##* }-pid${fixed_id}
echo $RES

fi

done