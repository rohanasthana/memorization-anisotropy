#!/usr/bin/env bash
set -euo pipefail

sd_versions=(1 2) #1 2 99(For Realistic Vision)
gen_numbers=(1) #1 4
guidance_scales=(7.5)
mode=("x,c|x") # "x,x|c" "x|c,c|x"
seeds=(51) #53 51 42
normalization=("None") # "L1" "L2" "None"

export CUDA_VISIBLE_DEVICES=0 

# Loop through each SD version (1 and 2)
for normalization_type in "${normalization[@]}"; do
    echo "Using normalization: $normalization_type"
    for seed in "${seeds[@]}"; do
        echo "Using seed: $seed"
        for sd_ver in "${sd_versions[@]}"; do
            # Conditional logic to set the correct data paths based on the SD version
            if [[ "$sd_ver" == "1" ]]; then
                data_paths=('prompts/sd1_mem.txt' 'prompts/sd1_nmem.txt')
                echo "Running for SD Version 1..."
            elif [[ "$sd_ver" == "2" ]]; then
                data_paths=('prompts/sd2_mem.txt' 'prompts/sd2_nmem.txt')
                echo "Running for SD Version 2..."
            elif [[ "$sd_ver" == "99" ]]; then
                data_paths=('prompts/RV_mem.txt' 'prompts/RV_nmem.txt')
                echo "Running for Realistic Vision..."
            else
                echo "Invalid sd_ver: $sd_ver"
                continue # Skip to the next iteration if the version is not 1 or 2
            fi

            # Loop through each data path file
            for mode_type in "${mode[@]}"; do
                for guidance_scale in "${guidance_scales[@]}"; do
                    for data_path in "${data_paths[@]}"; do
                        # Loop through each generation number
                        for gen_num in "${gen_numbers[@]}"; do
                            echo "--------------------------------------------------------"
                            echo "Executing: sd_ver=$sd_ver, data_path=$data_path, gen_num=$gen_num, mode=$mode_type"

                            python detect_mem.py \
                                --sd_ver "$sd_ver" \
                                --data_path "$data_path" \
                                --gen_num "$gen_num" \
                                --guidance_scale "$guidance_scale" \
                                --mode "$mode_type" \
                                --gen_seed "$seed" \
                                --normalization "$normalization_type"

                            echo "Execution complete."
                        done
                    done
                done
            done
        done
    done
done

echo "--------------------------------------------------------"
echo "All combinations have been run."