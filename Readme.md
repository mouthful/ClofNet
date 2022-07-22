# SE(3) Equivariant Graph Neural Networks with Complete Local Frames

Reference implementation in PyTorch of the equivariant graph neural network (**ClofNet**). You can find the paper [here](https://arxiv.org/abs/2110.14811). 

## Run the code

### Build environment
for newtonian system experiments
```   
conda create -n clof python=3.9 -y
conda activate clof
conda install -y -c pytorch pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.2 -y
```
for conformation generation task
```
conda install -y -c rdkit rdkit==2020.03.2.0
conda install -y scikit-learn pandas decorator ipython networkx tqdm matplotlib
conda install -y -c conda-forge easydict
pip install pyyaml
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric==1.6.3
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

### Conformation Generation
Equilibrium conformation generation targets on predicting stable 3D structures from 2D molecular graphs. Following [ConfGF](https://arxiv.org/abs/2105.03902), we evaluate the proposed ClofNet on the GEOM-QM9 and GEOM-Drugs datasets ([Axelrod & Gomez-Bombarelli, 2020](https://arxiv.org/abs/2006.05531)) as well as the ISO17 dataset ([Sch¨utt et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)). For the score-based generation framework, we build our algorithm based on the public codebase of [ConfGF](https://github.com/DeepGraphLearning/ConfGF). We sincerely thank their solid contribution for this field.

#### Dataset 
* **Offical Dataset**: The offical raw GEOM dataset is avaiable [[here]](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).

* **Preprocessed dataset**: We use the preprocessed datasets (GEOM, ISO17) published by ConfGF([[google drive folder]](https://drive.google.com/drive/folders/10dWaj5lyMY0VY4Zl0zDPCa69cuQUGb-6?usp=sharing)).

#### Train
```
cd confgen
python -u script/train.py --config_path ./config/qm9_clofnet.yml
```
#### Generation
```
python -u script/gen.py --config_path ./config/qm9_clofnet.yml --generator EquiGF --eval_epoch [epoch] --start 0 --end 100
```
#### Evaluation
```
python -u script/get_task1_results.py --input /root/to/generation --core 10 --threshold 0.5
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