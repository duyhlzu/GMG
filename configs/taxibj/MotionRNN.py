method = 'MotionRNN'
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
num_hidden = '128,128,128,128'
filter_size = 5
stride = 1
patch_size = 2
layer_norm = 0
# training
lr = 0.0003
batch_size = 16
sched = 'cosine'
warmup_epoch = 5

# python tools/train.py -d taxibj --epoch 200 -c configs/taxibj/MotionRNN.py --ex_name taxibj_motionrnn

# python tools/train.py -d taxibj --epoch 200 -c configs/taxibj/MotionRNN.py --ex_name taxibj_motionrnn_add_GGM

# python tools/train.py -d taxibj --epoch 200 -c configs/taxibj/MotionRNN.py --ex_name taxibj_motionrnn_add_GFM

# python tools/train.py -d taxibj --epoch 200 -c configs/taxibj/MotionRNN.py --ex_name taxibj_motionrnn_add_GFM_add_GGM_2

# python tools/train.py -d taxibj --epoch 200 -c configs/taxibj/MotionRNN.py --ex_name taxibj_motionrnn_add_flashback