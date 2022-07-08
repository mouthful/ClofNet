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
```
python -u main_nbody.py --max_training_samples 3000 --norm_diff True --LR_decay True --lr 0.005 --outf saved/nbody --data_mode small_20body --decay 0.4 --epochs 600 --exp_name small_20body --model evfn_norm --n_layers 4 --n_points 20 --data_toot "/casp/v-hezha1/workspace/EquiNODE/correct_data"
```
or

```
bash evfn4nbody.sh <model> <data_mode> <LR> <decay> <epoch> <exp_name> <n_layer> <n_points> <max_samples>
e.g.,
bash evfn4nbody.sh evfn_norm small_20body 0.005 0.4 600 small_20body 4 20 3000
```

## Cite
Please cite our papers if you use the model or this code in your own work:
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