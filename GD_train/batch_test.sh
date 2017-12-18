#CUDA_VISIBLE_DEVICES=2,4,5,6,7 python main.py --exp v3_random --resume ../Exps/v3_random/results/v3_random/004.ckpt --test 1 -b 50 -j 25
#CUDA_VISIBLE_DEVICES=2,4,5,6,7 python main_cel.py --exp v3_random_logit --resume ../Exps/v3_random_logit/results/random/019.ckpt --test 1 -b 50 -j 25

#CUDA_VISIBLE_DEVICES=2,4,5,6,7 python main_cel.py --exp v3_random --resume ../Exps/v3_random/results/v3_random_cel/010.ckpt --test 1 -b 50 -j 25
#CUDA_VISIBLE_DEVICES=3 python main_cel.py --exp v3_random --resume ../Exps/v3_random/results/v3_random_cel/010.ckpt --test 1 -b 20 -j 20
#CUDA_VISIBLE_DEVICES=3 python main.py --exp ensv3_random --test 1 -b 20 -j 20 --defense 0 --resume ../Exps/ensv3/results/20171022-201017/013.ckpt 
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --exp v3_random  -b 80 -j 40  --resume ../Exps/sample/results/20171022-215838/011.ckpt --save-dir tmp
#CUDA_VISIBLE_DEVICES=3 python main.py --exp resnet -b 20 -j 20 --test 1  

#CUDA_VISIBLE_DEVICES=3 python main.py --exp resnet -b 20 -j 20 --test 1 --defense 0
#CUDA_VISIBLE_DEVICES=3 python main.py --exp v3_random_logit -b 20 -j 20 --test 1 --resume ../Exps/v3_random_logit/results/random/011.ckpt
#CUDA_VISIBLE_DEVICES=1,3 python main_cel.py --exp rev_fgsm -b 20 -j 20 --test 1 --resume ../Exps/v3_random/results/v3_random_cel/010.ckpt
#CUDA_VISIBLE_DEVICES=1,3 python main_cel.py --exp rev_fgsm -b 20 -j 20 --test 1 --resume ../Exps/v3_random/results/v3_random_cel/010.ckpt --defense 0 

#CUDA_VISIBLE_DEVICES=6,7 python main.py --exp ens0 -b 40 -j 20 --test 1
#CUDA_VISIBLE_DEVICES=6,7 python main.py --exp ens2 -b 40 -j 20 --test 1
#CUDA_VISIBLE_DEVICES=6,7 python main.py --exp ens3 -b 40 -j 20 --test 1
#CUDA_VISIBLE_DEVICES=6,7 python main.py --exp ens4 -b 40 -j 20 --test 1
CUDA_VISIBLE_DEVICES=0,7 python main.py --exp ens-1 -b 40 -j 20 --test 1
CUDA_VISIBLE_DEVICES=0,7 python main.py --exp ens-2 -b 40 -j 20 --test 1
