export CUDA_VISIBLE_DEVICES=0

python main_msl.py --anormly_ratio 1 --num_epochs 3   --batch_size 64  --mode train --dataset MSL  --data_path MSL  --input_c 55 --output_c 55  --win_size 105  --patch_size 357
python main_msl.py --anormly_ratio 1  --num_epochs 10     --batch_size 64    --mode test    --dataset MSL   --data_path MSL --input_c 55    --output_c 55   --win_size 105  --patch_size 357

