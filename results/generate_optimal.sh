#!/bin/bash

cd logs
#ack -L "test filtered" | xargs rm
cd ..

mkdir -p results/wn18_optimal/ results/music_optimal/ results/yago_optimal/

./results/xshot-optimal-cli.py . logs/ucl_wn18_adv_xshot_v1/ logs/ucl_wn18_adv_xshot_logic_v1/
./results/aggregate_results.py
./results/tex_results_transposed.py
mv results/results* results/model_to_* results/wn18_optimal/

./results/xshot-optimal-cli.py . logs/ucl_music_adv_xshot_v1/ logs/ucl_music_adv_xshot_logic_v1/
./results/aggregate_results.py
./results/tex_results_transposed.py
mv results/results* results/model_to_* results/music_optimal/

./results/xshot-optimal-cli.py . logs/ucl_yago3_adv_xshot_v1/ logs/ucl_yago3_adv_xshot_logic_v1/
./results/aggregate_results.py
./results/tex_results_transposed.py
mv results/results* results/model_to_* results/yago_optimal/
