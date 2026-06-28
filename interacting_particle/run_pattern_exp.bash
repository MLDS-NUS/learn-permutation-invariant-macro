### 1.go into generate_dataset directory and generate simulation data and macro observation data
cd ./generate_dataset

# mode in [train, in_dst_test, diff_init_test, diff_N_test]. See Section 4.2 in the paper for details.
# E.g., to generate training data
python generate_simulation.py --mode train # will generate ./data/trajectories.npy, which is a [n_traj, T, n_particles, D] array
python generate_macro_obs_data.py --mode train # will generate ./data/macro_feature.npy

cd ../ # return to patterns/ directory

### 2.go back to patterns directory and run pattern experiments
python train_nflow_arqs.py # train our reconstruction model, using the generated ./data/trajectories.npy. Output will be saved in trained_nflow_gmm2/ directory by default.

### 3. After training, generate the Z data used for learning the ode
# mode in [train, in_dst_test, diff_init_test, diff_N_test].
python generate_dynamics_data.py --model_path trained_nflow_gmm2/exp1/ --mode train # output saved to evaluation_nflow_gmm2/exp1/macro_input/


### 4. Learn the dynamics using the generated Z data
cd learn_ode/
python train_example.py --data_path ../evaluation_nflow_gmm2/exp1/ # --data_path is the path to macro_input.
python test_ode.py # see the output figure in ../evaluation_nflow_gmm2/exp1/learned_dynamics/