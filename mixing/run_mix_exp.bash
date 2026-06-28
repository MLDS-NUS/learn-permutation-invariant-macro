### 1.go into generate_dataset directory and generate simulation data and macro observation data
cd ./generate_dataset/lmp_sim_files


## generate train and valid data

# particle mixing simulation in lammps
bash run_lmp_sim.sh 
# check the other bash files to generate test data, e.g., the test_left case in Sec. 4.3
bash run_lmp_sim_for_test_left.sh


cd .. # in generate_data/ folder
# generate micro trajectories
python load_lmp_trajectories.py --mode train # set --mode to in_dst_test, diff_dst_test, diff_N_test to generate test trajectories
python compute_pair_probs.py --mode train # set --mode to in_dst_test, diff_dst_test, diff_N_test to generate test macro features

cd .. # back to mixing/ folder

### 2. train our reconstruction model
python train_nflow_arqs.py # save the checkpoint to trained_nflow/exp1/ by default. Change the save_dir CLI to save to a different directory. Change the CLI args to train different models.

### 3. After training, generate the Z data used for learning the ode
python generate_dynamics_data.py --base_path ./trained_nflow/exp1/ --mode train  # suppose the trained model is saved in ./trained_nflow/exp1/. The output will be saved in <base_path>/dynamics_data_Z{z_dim}/. Change mode to generate test data.
# change --mode to the desired test mode, e.g., test_left,


### 4. Learn the dynamics using the generated Z data
cd learn_sde/
python learn_mixing_dynamics.py --data ../trained_nflow/exp1/dynamics_data_Z1/ --particle_type type1
python learn_mixing_dynamics.py --data ../trained_nflow/exp1/dynamics_data_Z1/ --particle_type type2

# test the learned dynamics. For example, left two panels in Fig. 6b in the paper
python plot_learned_dynamics.py --data ../trained_nflow/exp1/dynamics_data_Z1/macro_and_Z_types_test_left.npz --output_fig_name mean_std_left.png