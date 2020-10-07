target="/home/rynik/Software/anaconda3/envs/dpmd_gpu/lib/python3.8/site-packages/deepmd"

for f in common.py Trainer.py Data.py DataSystem.py
do
    scp ./source/train/$f kohn:$target/$f
done