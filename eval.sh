run_id=14
gpu=7
z_A=10
z_B=10
z_S=10

lamkl=1
beta1=0
beta2=0
beta3=0


CUDA_VISIBLE_DEVICES=${gpu} \
nohup python -u main.py \
--output_save_iter 1 --run_id ${run_id} \
--zA_dim ${z_A} --zB_dim ${z_B} --zS_dim ${z_S} \
--beta1 ${beta1} --beta2 ${beta2} --beta3 ${beta3} \
--ckpt_load_iter 70000 \
--dset_dir ../ --image_size 128 --dataset modalA --num_workers 2 --batch_size 64 \
--print_iter 50 \
--ckpt_save_iter 10000 \
--max_iter 70001 --lr_VAE 1e-4 --beta1_VAE 0.9 --beta2_VAE 0.999 --lamkl ${lamkl} \
--eval_metrics --eval_metrics_iter 100000 \
--viz_ll_iter 10 --viz_la_iter 50 --viz_port 8001 \
> ./log/eval${run_id}.${z_A}.${z_B}.${z_S}_lam${lamkl}.beta${beta1}.${beta2}.${beta3}.txt &
# 8001=cbb17
