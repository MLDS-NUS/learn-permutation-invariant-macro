#!/usr/bin/env bash

set -euo pipefail

save_dir=../dataset/lmp_dumps_test_right
mkdir -p "${save_dir}"
num_sims=100

# case "${save_dir}" in
#   ../dataset/lmp_dumps_test_left)
#     half_l=12
#     ;;
#   ../dataset/lmp_dumps_test_mid)
#     half_l=16
#     ;;
#   ../dataset/lmp_dumps_test_right)
#     half_l=20
#     ;;
#   *)
#     echo "Unsupported save_dir: ${save_dir}" >&2
#     exit 1
#     ;;
# esac
# mkdir -p "${save_dir}"

seed=12345
RANDOM="${seed}"
half_l=20

for i in $(seq 1 "${num_sims}"); do
  # vel_seed is a 30-bit integer in [0, 2^30 - 1].
  vel_seed=$(((RANDOM << 15) | RANDOM))
  mpirun -n 4 lmp -in sim_same_particle_number.in -screen none -log "${save_dir}/sim${i}.log" \
    -var half_l "${half_l}" -var dumpfile "${save_dir}/sim${i}.lammpstrj" -var vel_seed "${vel_seed}"
  echo "Completed simulation ${i}/${num_sims} with half_l=${half_l} vel_seed=${vel_seed}"
done
