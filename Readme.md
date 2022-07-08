# SE(3) Equivariant Graph Neural Networks with Complete Local Frames

Reference implementation in PyTorch of the equivariant graph neural network (**ClofNet**). You can find the paper [here](https://arxiv.org/abs/2110.14811). 

## Run the code

### Build environment
```   
conda create -n clof python=3.9 -y
conda activate clof
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch -y
```

### Newtonian many-body system

The code for many-body system modeling is highly inspired by [EGNN](https://github.com/vgsatorras/egnn). Many thanks to their solid contribution.
#### Create Many-body dataset
```
cd newtonian/dataset
bash script.sh
```

#### Run experiments
* for the ES(5) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.005 --outf saved/newtonian \
--data_mode small --decay 0.2 --epochs 600 --exp_name clof_vel_small_2body --model clof_vel --n_layers 4 --data_toot <root_of_data>
```
* for the ES(20) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.005 --outf saved/newtonian \
--data_mode small_20body --decay 0.2 --epochs 600 --exp_name clof_vel_small_2body --model clof_vel --n_layers 4 --data_toot <root_of_data>
```
* for the G+ES(20) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.005 --outf saved/newtonian \
--data_mode static_20body --decay 0.2 --epochs 600 --exp_name clof_vel_static_20body --model clof_vel --n_layers 4 --data_toot <root_of_data>
```
* for the L+ES(20) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.005 --outf saved/newtonian \
--data_mode dynamic_20body --decay 0.2 --epochs 600 --exp_name clof_vel_static_20body --model clof_vel --n_layers 4 --data_toot <root_of_data>
```

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{weitao_clofnet_2021,
  title     = {{SE(3)} Equivariant Graph Neural Networks with Complete Local Frames},
  author =  {Weitao Du and
            He Zhang and
            Yuanqi Du and
            Qi Meng and
            Wei Chen and
            Nanning Zheng and
            Bin Shao and
            Tie{-}Yan Liu},
  booktitle={International Conference on Machine Learning, {ICML} 2022, 17-23 July
               2022, Baltimore, Maryland, {USA}},
  year = {2021}
}
```