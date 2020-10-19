"""
Code to warm start the correspondence prediction network with small amount of supervised data.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""

import os
from os.path import split, join, exists
from data_loader.data_loader import MyDataLoader
import torch
from models.part_specific_pointnet import PointNet2Part
from models.trainer import TrainerPartSpecificNet


def main(mode, exp_id, optimizer, batch_size, epochs, save_name=None, num_saves=None, augment=False, naked=False,
         split_file=None):
    if split_file is None:
        split_file = 'assets/data_split_01.pkl'

    corr_net = PointNet2Part(in_features=0, num_parts=14, num_classes=3)

    if naked:
        exp_name = 'part_specific_net/naked_exp_id_{}'.format(exp_id)
    else:
        exp_name = 'part_specific_net/exp_id_{}'.format(exp_id)

    if mode == 'train':
        dataset = MyDataLoader('train', batch_size, num_workers=16, augment=augment, naked=naked,
                                 split_file=split_file).get_loader()
        trainer = TrainerPartSpecificNet(corr_net, torch.device("cuda"), dataset, None, exp_name,
                                         optimizer=optimizer)
        trainer.train_model(epochs)
    elif mode == 'val':
        dataset = MyDataLoader('val', batch_size, num_workers=16, naked=naked,
                                 split_file=split_file).get_loader(shuffle=False)
        trainer = TrainerPartSpecificNet(corr_net, torch.device("cuda"), None, dataset, exp_name,
                                         optimizer=optimizer)
        trainer.pred_model(save_name=save_name, num_saves=num_saves)
    elif mode == 'eval':
        dataset = MyDataLoader('val', batch_size, num_workers=16, naked=naked,
                                 split_file=split_file).get_loader(shuffle=False)
        trainer = TrainerPartSpecificNet(corr_net, torch.device("cuda"), None, dataset, exp_name,
                                         optimizer=optimizer)
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
    parser.add_argument('-augment', default=False, action='store_true')
    # Train network for dressed or undressed scans
    parser.add_argument('-naked', default=False, action='store_true')
    parser.add_argument('-split_file', type=str, default=None)
    # Validation specific arguments
    parser.add_argument('-mode', default='train', choices=['train', 'val', 'eval'])
    parser.add_argument('-save_name', default='', type=str)
    parser.add_argument('-num_saves', default=None, type=int)
    args = parser.parse_args()

    if args.mode == 'val':
        assert len(args.save_name) > 0
        main('val', args.exp_id, args.optimizer, args.batch_size, args.epochs,
             save_name=args.save_name, num_saves=args.num_saves, naked=args.naked, split_file=args.split_file)
    elif args.mode == 'train':
        main('train', args.exp_id, args.optimizer, args.batch_size, args.epochs, augment=args.augment, naked=args.naked,
             split_file=args.split_file)

    elif args.mode == 'eval':
        main('eval', args.exp_id, args.optimizer, args.batch_size, args.epochs, augment=args.augment, naked=args.naked,
             split_file=args.split_file)
