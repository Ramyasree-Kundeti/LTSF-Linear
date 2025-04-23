
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear

python -u run_longExp.py \
  --is_training 1 \
  --model_id ushcn_$seq_len'_'96 \
  --model $model_name \
  --data ushcn \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --fold 0 \
  --ot 36  \
  --ft 12  \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'$seq_len'_'96.log