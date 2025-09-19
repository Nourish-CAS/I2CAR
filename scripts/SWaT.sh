export CUDA_VISIBLE_DEVICES=0

python main_psm.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset SWAT  --data_path SWAT --input_c 51    --output_c 51  --loss_fuc MSE  --win_size 105  --patch_size 357
python main_psm.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset SWAT   --data_path SWAT  --input_c 51    --output_c 51  --loss_fuc MSE  --win_size 105  --patch_size 357

