# -------------------------------------
# model_variant: evfn, evfn_naive, evfn_raw_basis, evfn_1basis, evfn_nbasis
# data_mode: small, static, dynamic
# date_tag: running date

if [ $# != 9 ]; then
    echo "Usage $0 <model> <data_mode> <LR> <decay> <epoch> <exp_name> <n_layer> <n_points> max_samples"
    exit 1
fi
model=$1
data_mode=$2
LR=$3
decay=$4
epoch=$5
exp_name=$6
n_layer=$7
n_points=$8
max_samples=$9

# python -u main_nbody.py --max_training_samples ${max_samples} --norm_diff True --LR_decay True --lr ${LR} --outf saved/${date} --data_mode ${data_mode} --decay ${decay} --epochs ${epoch} --exp_name ${exp_name} --model ${model} --n_layers ${n_layer} --n_points 20

python -u main_nbody.py --max_training_samples $max_samples --norm_diff True --LR_decay True --lr ${LR} --outf saved/nbody --data_mode ${data_mode} --decay ${decay} --epochs ${epoch} --exp_name ${exp_name} --model ${model} --n_layers ${n_layer} --n_points ${n_points}
