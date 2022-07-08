saved_dir="data" # the path of output directory

'''
suffix: the suffix of generated dataset (e.g., small_20body)
simulation: the mode of force field
    1. charged: electronstatic force
    2. static: gravity + electronstatic force
    3. dynamic: lorentz + electronstatic force
    4. fixcharge: there exists a particle with large charge located at a fixed position
n_balls: system size
num_train: number of training trajectories
other parameters: see generate_dataset.py
'''

suffix=small_20body
simulation=charged
n_balls=20
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}

# suffix=static
suffix=static_20body
simulation=static
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}

# suffix=dynamic
suffix=dynamic_20body
simulation=dynamic
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}


n_balls=5
suffix=fixcharge
simulation=fixcharge
python -u generate_dataset.py --num-train 3000 --seed 43 --sufix ${suffix} --saved_dir ${saved_dir} --simulation ${simulation} --n_balls ${n_balls}