GPU_ID=0

dataset=seq-tinyimg
method=AT
wandb_project=CIL

alpha=1
beta=1
gamma=0

buffer=2000  # no-buffer: set buffer=0
model=flair  # no-buffer: set model=flair_nobuffer
aug=none

optim_wd=1e-05
epochs=100
batch_size=128
lr=1

for aug in none ra aua
do
    CUDA_VISIBLE_DEVICES=$GPU_ID python  ./utils/main.py --model $model --robust_method $method --wandb_name ${model}_${alpha}_${beta}_${gamma}_${aug}_${epochs}_${lr}_${batch_size} --wandb_tags tinyImageNet --n_epochs $epochs --dataset $dataset --buffer_size $buffer --lr $lr --optim_wd $optim_wd --alpha $alpha --beta $beta --gamma $gamma --nowand 0 --batch_size $batch_size --wandb_project $wandb_project --aug $aug
done
