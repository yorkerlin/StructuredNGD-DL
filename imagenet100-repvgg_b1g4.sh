#source  /scratch-ssd/wlin/mambaforge/etc/profile.d/conda.sh
conda activate pytorch-new

batch_size="128"
update_freq=60 #only for KFAC and Our method (local)
run_id=000
damping=0.0005 #only for KFAC and Our method (local)
data_name="imagenet100"
model_name="repvgg_b1g4"


lr=0.0025
opt_name="local"
weight_decay=0.01 
python main.py --dataset $data_name --optimizer $opt_name --network $model_name \
    --batch_size $batch_size --epoch 120 --milestone 40,80 --learning_rate $lr \
    --weight_decay $weight_decay --damping $damping \
    --TCov $update_freq --TInv $update_freq --faster 1 --run_id $run_id


lr=0.001
opt_name="kfac"
weight_decay=0.01 
python main.py --dataset $data_name --optimizer $opt_name --network $model_name \
    --batch_size $batch_size --epoch 120 --milestone 40,80 --learning_rate $lr \
    --weight_decay $weight_decay --damping $damping \
    --TCov $update_freq --TInv $update_freq --faster 1 --run_id $run_id

lr=0.001
opt_name="adamw"
weight_decay=0.01 
python main.py --dataset $data_name --optimizer $opt_name --network $model_name \
    --batch_size $batch_size --epoch 120 --milestone 40,80 --learning_rate $lr \
    --weight_decay $weight_decay --damping $damping \
    --TCov $update_freq --TInv $update_freq --faster 1 --run_id $run_id


lr=0.0001
opt_name="lion"
weight_decay=0.1 
python main.py --dataset $data_name --optimizer $opt_name --network $model_name \
    --batch_size $batch_size --epoch 120 --milestone 40,80 --learning_rate $lr \
    --weight_decay $weight_decay --damping $damping \
    --TCov $update_freq --TInv $update_freq --faster 1 --run_id $run_id


lr=0.03
opt_name="sgd"
weight_decay=0.001 
python main.py --dataset $data_name --optimizer $opt_name --network $model_name \
    --batch_size $batch_size --epoch 120 --milestone 40,80 --learning_rate $lr \
    --weight_decay $weight_decay --damping $damping \
    --TCov $update_freq --TInv $update_freq --faster 1 --run_id $run_id

lr=0.0002
opt_name="adam"
weight_decay=0.001 
python main.py --dataset $data_name --optimizer $opt_name --network $model_name \
    --batch_size $batch_size --epoch 120 --milestone 40,80 --learning_rate $lr \
    --weight_decay $weight_decay --damping $damping \
    --TCov $update_freq --TInv $update_freq --faster 1 --run_id $run_id


