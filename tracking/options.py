from collections import OrderedDict

opts = OrderedDict()
# ALL about
opts['gpu'] = '0'
opts['debug'] = True
opts['hard_min'] = True
opts['vis'] = True
opts['seq_name'] = "Bird1"
opts['seed'] = 1234

# input size
opts['img_size'] = 64
opts['padding'] = 0

# batch size
opts['batch_extractor'] = 256
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_pos_cand'] = 256
opts['batch_test'] = 256

opts['batch_test_neg'] = 256
opts['batch_test_pos'] = 256

# candidates sampling
opts['n_samples'] = 256  
opts['trans'] = 0.6
opts['scale'] = 1.2        
opts['trans_limit'] = 1.5

# training examples sampling
opts['trans_pos'] = 0.1
opts['scale_pos'] = 1.3
opts['trans_neg_init'] = 1
opts['scale_neg_init'] = 1.6
opts['trans_neg'] = 2
opts['scale_neg'] = 1.3

# bounding box regression
opts['n_bbreg'] = 5000
opts['overlap_bbreg'] = [0.6, 1]
opts['trans_bbreg'] = 0.3
opts['scale_bbreg'] = 1.6
opts['aspect_bbreg'] = 1.1

# initial training
opts['lr_init'] = 0.001  # 0.0005
opts['init_freq'] = 50  # Init frequency
opts['update_freq'] = 30  # Update frequency
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

# online training
opts['lr_update'] = 0.001 # 0.001
opts['n_pos_update'] = 200    # 500      # 200  # 50
opts['n_neg_update'] = 400   # 2000      # 800  # 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

# update criteria
opts['long_interval'] = 10  # 15

# Sample space samples number
opts["samplespace_pos"] = 4             # 30        # 20
opts["samplespace_neg"] = 10         # 10

# for indicating online update samples
opts['n_frames_long'] = 30
opts['n_frames_update'] = 30

opts['success_thr'] = 0.2

# training
# opts['grad_clip'] = 10

# classifier's structure
opts['in_dim'] = 2048  # 256
opts['l_dim'] = 2048

# delay
opts['delay_freq'] = 1
opts['score_ratio'] = 1  # classifier's score weight

# opts["ckpt_path"] = "G:/workspace/code/tracking/checkpoints/DCLT/2048/last"  # '/home/xulong/xx/workspace/checkpoint/420000_128_128_2.2'
opts["ckpt_path"] = "../net/weights/last"  # '/home/xulong/xx/workspace/checkpoint/420000_128_128_2.2'

# print('-----')
# print('args')
# print('-----')
# for k, v in opts.items():
#     print('{}: {}'.format(k, v))
# print('-----')
