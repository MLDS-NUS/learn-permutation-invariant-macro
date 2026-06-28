#!/usr/bin/env bash

set -euo pipefail

save_dir=../dataset/lmp_dumps_inDistribution_test
mkdir -p "${save_dir}"

num_sims=200
# half_l is sampled as a float in [min_half_l, max_half_l].
min_half_l=10
max_half_l=22

seed=67890
chunk_count=10
if ((chunk_count > num_sims)); then
  chunk_count="${num_sims}"
fi

format_seconds() {
  local total=$1
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02d:%02d:%02d" "${h}" "${m}" "${s}"
}

run_chunk() {
  local chunk_id=$1
  local start_idx=$2
  local end_idx=$3
  local chunk_seed=$4
  local max_vel_seed=2147483647

  RANDOM="${chunk_seed}"
  local start_ts
  start_ts=$(date +%s)
  local total=$((end_idx - start_idx + 1))
  local completed=0

  for i in $(seq "${start_idx}" "${end_idx}"); do
    completed=$((completed + 1))
    # Combine two 15-bit RANDOM draws into one 30-bit value for uniform float sampling.
    rand=$(((RANDOM << 15) | RANDOM))
    half_l=$(awk -v min="${min_half_l}" -v max="${max_half_l}" -v r="${rand}" \
      'BEGIN {printf "%.6f", min + (max - min) * (r / 1073741823)}')
    # Keep vel_seed within common LAMMPS 32-bit positive seed range.
    vel_seed=$(( (seed + i) % max_vel_seed ))
    if ((vel_seed == 0)); then
      vel_seed=1
    fi
    lmp -in sim_same_particle_number.in -screen none -log "${save_dir}/sim${i}.log" \
      -var half_l "${half_l}" -var dumpfile "${save_dir}/sim${i}.lammpstrj" -var vel_seed "${vel_seed}"
    if ((completed % 10 == 0 || completed == total)); then
      echo "Chunk ${chunk_id} completed simulation ${i}/${num_sims} with half_l=${half_l} vel_seed=${vel_seed}"

      now=$(date +%s)
      elapsed=$((now - start_ts))
      avg=$(awk -v e="${elapsed}" -v n="${completed}" 'BEGIN{print (n>0)? e/n : 0}')
      remaining=$(awk -v a="${avg}" -v left="$((total - completed))" 'BEGIN{print int(a*left)}')
      elapsed_fmt=$(format_seconds "${elapsed}")
      remaining_fmt=$(format_seconds "${remaining}")
      printf "Chunk %d progress %d/%d (global %d/%d) | elapsed %s | ETA %s\n" \
        "${chunk_id}" "${completed}" "${total}" "${i}" "${num_sims}" "${elapsed_fmt}" "${remaining_fmt}"
    fi
  done
}

base=$((num_sims / chunk_count))
rem=$((num_sims % chunk_count))
for chunk_id in $(seq 1 "${chunk_count}"); do
  start_idx=$(((chunk_id - 1) * base + 1))
  end_idx=$((chunk_id * base))
  if ((chunk_id == chunk_count)); then
    end_idx=$((end_idx + rem))
  fi
  run_chunk "${chunk_id}" "${start_idx}" "${end_idx}" "$((seed + chunk_id))" &
done

wait
