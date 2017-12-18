CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py --exp ae --lr 0.001 -j 40 -b 80 --resume ../pixel_Exps/ae/results/random/002.ckpt --save-dir random --test 1
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py --exp d_unet --lr 0.001 -j 40 -b 80 --resume ../pixel_Exps/d_unet/results/random/002.ckpt --save-dir random --test 1
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python main.py --exp da_unet --lr 0.001 -j 40 -b 80 --resume ../pixel_Exps/da_unet/results/random/006.ckpt --save-dir random --test 1
