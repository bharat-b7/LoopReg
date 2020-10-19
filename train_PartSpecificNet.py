"""
Train the network with self-supervision. This code assumes that you have already performed the supervised warm start.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""

import os
from os.path import split, join, exists
from glob import glob
import numpy as np
from data_loader.data_loader import MyDataLoaderCacher
import torch
from models.part_specific_pointnet import PointNet2Part
from models.trainer import CombinedTrainer

def load_pretrained(path, net):
    checkpoints = glob(path + '/*')
    if len(checkpoints) == 0:
        print('No checkpoints found at {}'.format(path))
        return 0

    checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
    checkpoints = np.array(checkpoints, dtype=int)
    checkpoints = np.sort(checkpoints)
    path = join(path, 'checkpoint_epoch_{}.tar'.format(checkpoints[-1]))
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Loaded weights from ', path)
    return

def main(mode, exp_id, optimizer, batch_size, epochs, pretrained_path=None, save_name=None, num_saves=None,
         naked=False, cache_suffix='cache', checkpoint_number=-1, split_file=None):

    if split_file is None:
        split_file = 'assets/data_split_01.pkl'

    corr_net = PointNet2Part(in_features=0, num_parts=14, num_classes=3)

    if pretrained_path is not None:
        # Load weights from pre-training
        load_pretrained(pretrained_path, corr_net)
    else:
        print('Not initializing with pre-trained supervised correspondence network')

    if naked:
        exp_name = 'combined_net/naked_exp_id_{}'.format(exp_id)
    else:
        exp_name = 'combined_net/exp_id_{}'.format(exp_id)

    if mode == 'train':
        dataset = MyDataLoaderCacher('train', batch_size, cache_suffix=cache_suffix,
                                     split_file=split_file,
                                     num_workers=30, augment=False, naked=naked).get_loader()
        """ 
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        trainer = CombinedTrainer(corr_net, torch.device("cuda"), dataset, None, exp_name,
                                  optimizer=optimizer, opt_dict={'cache_folder': cache_suffix,
                                                                 'iter_per_step': {1: 100, 2: 100, 3: 1},  # per phase
                                                                 'epochs_phase_01': 1, 'epochs_phase_02': 2},
                                  checkpoint_number=checkpoint_number)
        trainer.train_model(epochs)
    elif mode == 'val':
        # MyDataLoaderFaustTest, MyDataLoaderCacher
        dataset = MyDataLoaderCacher('val', batch_size, cache_suffix=cache_suffix, split_file=split_file,
                                     num_workers=16, naked=naked).get_loader(shuffle=False)
        trainer = CombinedTrainer(corr_net, torch.device("cuda"), None, dataset, exp_name,
                                  optimizer=optimizer, opt_dict={'cache_folder': cache_suffix,
                                                                 'iter_per_step': {1: 200, 2: 101, 3: 1}},
                                  checkpoint_number=checkpoint_number)
        trainer.fit_test_sample(save_name=save_name, num_saves=num_saves)

    elif mode == 'eval':
        dataset = MyDataLoaderCacher('val', batch_size, cache_suffix=cache_suffix, split_file=split_file,
                                     num_workers=16, naked=naked).get_loader(shuffle=False)
        trainer = CombinedTrainer(corr_net, torch.device("cuda"), None, dataset, exp_name,
                                  optimizer=optimizer, opt_dict={'cache_folder': cache_suffix},
                                  checkpoint_number=checkpoint_number)
        trainer.eval_model('val')
    else:
        print('Invalid mode. should be either train, val or eval.')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('exp_id', type=str)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-optimizer', default='Adam', type=str)
    parser.add_argument('-epochs', default=150, type=int)
    # parser.add_argument('-augment', default=False, action='store_true')
    parser.add_argument('-naked', default=False, action='store_true')
    parser.add_argument('-pretrained_path', type=str, default=None)
    parser.add_argument('-cache_suffix', type=str, default='cache')
    parser.add_argument('-checkpoint_number', default=-1, type=int)
    parser.add_argument('-split_file', type=str, default=None)
    # Validation specific arguments
    parser.add_argument('-mode', default='train', choices=['train', 'val', 'eval'])
    parser.add_argument('-save_name', default='', type=str)
    parser.add_argument('-num_saves', default=None, type=int)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # args = lambda : None
    # args.exp_id = '2'
    # args.batch_size = 2
    # args.epochs = 3
    # args.naked = True
    # args.augment = False
    # args.pretrained_path = '/BS/bharat-2/work/LearntRegistration/experiments/part_specific_net/naked_exp_id_01/checkpoints'
    # args.mode = 'val'
    # args.optimizer = 'Adam'
    # args.save_name = 'corr'
    # args.num_saves = 2

    if args.mode == 'val':
        assert len(args.save_name) > 0
        main('val', args.exp_id, args.optimizer, args.batch_size, args.epochs,
             save_name=args.save_name, num_saves=args.num_saves, naked=args.naked,
             pretrained_path=args.pretrained_path, checkpoint_number=args.checkpoint_number, split_file=args.split_file)
    elif args.mode == 'train':
        main('train', args.exp_id, args.optimizer, args.batch_size, args.epochs, naked=args.naked,
             pretrained_path=args.pretrained_path, cache_suffix=args.cache_suffix,
             checkpoint_number=args.checkpoint_number, split_file=args.split_file)

    elif args.mode == 'eval':
        main('eval', args.exp_id, args.optimizer, args.batch_size, args.epochs, naked=args.naked,
             checkpoint_number=args.checkpoint_number, split_file=args.split_file)

"""
[TRAIN]
python train_PartSpecificNet.py 1 -batch_size 16 -pretrained_path experiments/part_specific_net/naked_exp_id_12/checkpoints/ \
-cache_suffix new_cache_5 -split_file assets/data_split_01.pkl 

[TEST]
python train_PartSpecificNet.py 1 -batch_size 12 -split_file assets/data_split_01.pkl -mode val -save_name corr -naked
"""