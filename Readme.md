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

* This task is inspired by (Kipf et al., 2018; Fuchs et al., 2020; Satorras et al., 2021b), where a 5-body charged system is controlled by the electrostatic force field. Note that the force direction between any two particles is always along the radial direction in the original setting. To validate the effectiveness of ClofNet on modeling arbitrary force directions, we also impose two external force fields into the original system, a gravity field and a Lorentz-like dynamical force field, which can provide more complex and dynamical force directions.
* The original source code for generating trajectories comes from [Kipf et al., 2018](https://github.com/ethanfetaya/NRI) and is modified by [EGNN](https://github.com/vgsatorras/egnn). We further extend the version of EGNN to three new settings, as described in Section 7.1 of our [paper](https://arxiv.org/abs/2110.14811). We sincerely thank the solid contribution of these two works. 

#### Create Many-body dataset
```
cd newtonian/dataset
bash script.sh
```

#### Run experiments
* for the ES(5) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.01 --outf saved/newtonian \
--data_mode small --decay 0.9 --epochs 400 --exp_name clof_vel_small_5body --model clof_vel --n_layers 4 --data_root <root_of_data>
```
* for the ES(20) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.01 --outf saved/newtonian \
--data_mode small_20body --decay 0.9 --epochs 600 --exp_name clof_vel_small_20body --model clof_vel --n_layers 4 --data_root <root_of_data>
```
* for the G+ES(20) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.01 --outf saved/newtonian \
--data_mode static_20body --decay 0.9 --epochs 200 --exp_name clof_vel_static_20body --model clof_vel --n_layers 4 --data_root <root_of_data>
```
* for the L+ES(20) setting, run
```
python -u main_newtonian.py --max_training_samples 3000 --norm_diff True --LR_decay True --decay 0.9 --lr 0.01 --outf saved/newtonian \
--data_mode dynamic_20body --epochs 600 --exp_name clof_vel_dynamic_20body --model clof_vel --n_layers 4 --data_root <root_of_data>
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