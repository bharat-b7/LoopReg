"""
Code to train the network.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""
from __future__ import division
from os.path import join, split, exists
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import time
import trimesh
import pickle as pkl
import numpy as np
from collections import Counter
from tqdm import tqdm
from kaolin.metrics.point import SidedDistance
from models.volumetric_SMPL import VolumetricSMPL
from lib.smpl_paths import SmplPaths
from lib.th_smpl_prior import get_prior
from lib.torch_functions import batch_gather, chamfer_distance

NUM_POINTS = 30000

def closest_index(src_points: torch.Tensor, tgt_points: torch.Tensor):
    """
    Given two point clouds, finds closest vertex id
    :param src_points: B x N x 3
    :param tgt_points: B x M x 3
    :return B x N
    """
    sided_minimum_dist = SidedDistance()
    closest_index_in_tgt = sided_minimum_dist(
            src_points, tgt_points)
    return closest_index_in_tgt

class Trainer(object):
    '''
    Trainer for predicting scan to SMPL correspondences from a pointcloud.
    This trainer does not optimise the correspondences.
    '''

    def __init__(self, model, device, train_loader, val_loader, exp_name, optimizer='Adam'):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_data_loader = train_loader
        self.val_data_loader = val_loader

        self.exp_path = join(os.path.dirname(__file__), '../experiments/{}'.format(exp_name))
        self.checkpoint_path = join(self.exp_path, 'checkpoints/')
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(join(self.exp_path, 'summary'.format(exp_name)))
        self.val_min = None

    @staticmethod
    def sum_dict(los):
        temp = 0
        for l in los:
            temp += los[l]
        return temp

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss_ = self.compute_loss(batch)
        loss = self.sum_dict(loss_)
        loss.backward()
        self.optimizer.step()

        return {k: loss_[k].item() for k in loss_}

    def compute_loss(self, batch):
        device = self.device
        scan = batch.get('scan').to(device)
        smpl = batch.get('smpl').to(device)
        logits = self.model(scan)

        # MSE loss on vertices
        mse = F.mse_loss(logits.view(-1, 6890, 3), smpl)

        loss = {'mse': mse}
        return loss

    def train_model(self, epochs):
        start = self.load_checkpoint()
        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))

            if epoch % 1 == 0:
                self.save_checkpoint(epoch)
                '''Add validation loss here if over-fitting is suspected'''

            sum_loss = None
            loop = tqdm(self.train_data_loader)
            for n, batch in enumerate(loop):
                loss = self.train_step(batch)
                # print(" Epoch: {}, Current loss: {}".format(epoch, loss))
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss += Counter(loss)
                l_str = 'Ep: {}'.format(epoch)
                for l in sum_loss:
                    l_str += ', {}: {:0.5f}'.format(l, sum_loss[l] / (1 + n))
                loop.set_description(l_str)

            for l in sum_loss:
                self.writer.add_scalar(l, sum_loss[l] / len(self.train_data_loader), epoch)

    def save_checkpoint(self, epoch):
        path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(epoch))
        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self, number=-1):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        if number == -1:
            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)

            if checkpoints[-1] == 0:
                print('Not loading model as this is the first epoch')
                return 0

            path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(checkpoints[-1]))
        else:
            path = join(self.checkpoint_path, 'checkpoint_epoch_{}.tar'.format(number))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

class TrainerPartSpecificNet(Trainer):
    def compute_loss(self, batch):
        device = self.device
        scan = batch.get('scan').to(device)
        part_label = batch.get('part_labels').to(device).long()
        correspondences = batch.get('correspondences').to(device)

        logits = self.model(scan)
        ce = F.cross_entropy(logits['part_labels'], part_label.squeeze(-1))
        mse = F.mse_loss(logits['correspondences'].permute(0, 2, 1), correspondences)

        loss = {'cross_entropy': ce, 'correspondences': mse}
        return loss

    def eval_model(self, mode):
        """
        average accuracy and distance
        """
        epoch = self.load_checkpoint()
        print('Evaluating on {} set with epoch {}'.format(mode, epoch))

        if mode == 'train':
            data_loader = self.train_data_loader
        else:
            data_loader = self.val_data_loader

        correct, total, error = 0, 0, 0
        for batch in tqdm(data_loader):
            scan = batch.get('scan').to(self.device)
            label_gt = batch.get('part_labels').view(-1, )
            corr_gt = batch.get('correspondences')
            with torch.no_grad():
                pred = self.model(scan)

                _, predicted = torch.max(pred['part_labels'].data, 1)
                correct += (predicted.cpu().view(-1, ) == label_gt).sum().item()
                total += label_gt.shape[0]

                # ToDo: map correspondences in R^3 to SMPL surface

                error += (((corr_gt - pred['correspondences'].permute(0, 2, 1).cpu())**2).sum(dim=-1)**0.5).mean()

        print('Part Accuracy: {}, Corr. Dist: {}'.format(correct * 1. / total, error/len(data_loader)))

    def pred_model(self, save_name, num_saves=None):
        """
        :param save_name: folder name to save the results to inside the experiment folder
        :param num_saves: number of examples to save, None implies save all
        """
        from psbody.mesh import Mesh
        from os.path import join, exists
        import os

        self.model.train(False)

        epoch = self.load_checkpoint()
        print('Testing with epoch {}'.format(epoch))
        if not exists(join(self.exp_path, save_name + '_ep_{}'.format(epoch))):
            os.makedirs(join(self.exp_path, save_name + '_ep_{}'.format(epoch)))

        count = 0
        for batch in tqdm(self.val_data_loader):
            names = batch.get('name')
            scan = batch.get('scan')
            vcs = batch.get('scan_vc').numpy()
            with torch.no_grad():
                out = self.model(scan.to(self.device))
                pred = out['correspondences'].detach().permute(0, 2, 1).cpu().numpy()
                part_label = out['part_labels'].detach()
                _, part_label = torch.max(part_label.data, -1)
                part_label = np.array(part_label.cpu())

            for v, name, sc, vc, pl in zip(pred, names, scan, vcs, part_label):
                name = split(name)[1]
                # save scan with vc
                t = Mesh(sc, [])
                t.set_vertex_colors(vc)
                t.write_ply(join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_scan.ply'))

                # save raw correspondences
                t = Mesh(v, [])
                t.set_vertex_colors(vc)
                t.write_ply(join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_corr_raw.ply'))

                # save part_labels
                t.set_vertex_colors_from_weights(pl)
                t.write_ply(join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_part.ply'))

                count += 1
            if (num_saves is not None) and (count > num_saves):
                break


class CombinedTrainer(Trainer):
    """
    Trainer to jointly optimize correspondence predictor network and instance specific SMPL parameters
    """

    def __init__(self, model, device, train_loader, val_loader, exp_name, opt_dict={}, optimizer='Adam',
                 checkpoint_number=-1, train_supervised=False):
        """
        :param model: correspondence prediction network
        :param device: cuda or cpu
        :param train_loader:
        :param val_loader:
        :param exp_name:
        :param opt_dict: dict containing optimization specific parameteres
        :param optimizer:
        :param checkpoint_number: load a specific checkpoint, -1 => load latest
        :param train_supervised:
        """
        self.model = model.to(device)
        self.device = device
        self.opt_dict = self.parse_opt_dict(opt_dict)
        self.optimizer_type = optimizer
        self.optimizer = self.init_optimizer(optimizer, self.model.parameters(), learning_rate=0.001)

        self.train_data_loader = train_loader
        self.val_data_loader = val_loader
        self.train_supervised = train_supervised
        self.checkpoint_number = checkpoint_number

        # Load vsmpl
        self.vsmpl = VolumetricSMPL('/BS/bharat-2/work/LearntRegistration/test_data/volumetric_smpl_function_64',
                                    device, 'male')
        sp = SmplPaths(gender='male')
        self.ref_smpl = sp.get_smpl()
        self.template_points = torch.tensor(
            trimesh.Trimesh(vertices=self.ref_smpl.r, faces=self.ref_smpl.f).sample(NUM_POINTS).astype('float32'),
            requires_grad=False).unsqueeze(0)

        self.pose_prior = get_prior('male', precomputed=True)

        # Load smpl part labels
        with open('/BS/bharat-2/work/LearntRegistration/test_data/smpl_parts_dense.pkl', 'rb') as f:
            dat = pkl.load(f, encoding='latin-1')
        self.smpl_parts = np.zeros((6890, 1))
        for n, k in enumerate(dat):
            self.smpl_parts[dat[k]] = n

        self.exp_path = join(os.path.dirname(__file__), '../experiments/{}'.format(exp_name))
        self.checkpoint_path = join(self.exp_path, 'checkpoints/'.format(exp_name))
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(join(self.exp_path, 'summary'.format(exp_name)))
        self.val_min = None

    @staticmethod
    def init_optimizer(optimizer, params, learning_rate=1e-4):
        if optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=learning_rate, betas=(0.9, 0.999))
        if optimizer == 'Adadelta':
            optimizer = optim.Adadelta(params)
        if optimizer == 'RMSprop':
            optimizer = optim.RMSprop(params, momentum=0.9)
        return optimizer

    @staticmethod
    def parse_opt_dict(opt_dict):
        timestamp = int(time.time())
        parsed_dict = {'iter_per_step': {1: 200, 2: 200, 3: 1}, 'cache_folder': join('cache', str(timestamp)),
                       'epochs_phase_01': 0, 'epochs_phase_02': 0}
        """ 
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        for k in parsed_dict:
            if k in opt_dict:
                if k == 'cache_folder':
                    parsed_dict[k] = join(opt_dict[k], str(timestamp))
                else:
                    parsed_dict[k] = opt_dict[k]
        print('Cache folder: ', parsed_dict['cache_folder'])
        return parsed_dict

    @staticmethod
    def get_optimization_weights(phase):
        """
        Phase_01: Initialised SMPL are far off from the solution. Optimize SMPL based on correspondences.
        Phase_02: SMPL models are close to solution. Fit SMPL based on ICP.
        Phase_03: Jointly update SMPL and correspondences.
        """
        if phase == 1:
            return {'corr': 2 * 10. ** 2, 'templ': 2 * 10. ** 2, 's2m': 10. ** 1, 'm2s': 10. ** 1, 'pose_pr': 10. ** -2,
                    'shape_pr': 10. ** -1}
        elif phase == 2:
            return {'corr': 10. ** 0, 'templ': 2 * 10. ** 2, 's2m': 2 * 10. ** 3, 'm2s': 10. ** 3, 'pose_pr': 10. ** -4,
                    'shape_pr': 10. ** -1}
        else:
            return {'corr': 2 * 10. ** 2, 'templ': 2 * 10. ** 2, 's2m': 10. ** 4, 'm2s': 10. ** 4, 'pose_pr': 10. ** -4,
                    'shape_pr': 10. ** -1}

    def train_model(self, epochs):
        start = self.load_checkpoint(number=self.checkpoint_number)
        phase = -1
        for epoch in range(start, epochs):
            print('Start epoch {}'.format(epoch))

            if epoch % 1 == 0:
                self.save_checkpoint(epoch)
                '''Add validation loss here if over-fitting is suspected'''

            # parse training phase
            if epoch < self.opt_dict['epochs_phase_01'] and phase != 1:
                phase = 1
                print('Starting training phase 1')
            elif (epoch < (self.opt_dict['epochs_phase_02'] + self.opt_dict['epochs_phase_01'])) and phase < 2:
                print('Starting training phase 2')
                phase = 2
                # get a new cache folder
                self.opt_dict['cache_folder'] = split(self.opt_dict['cache_folder'])[0]
                self.opt_dict = self.parse_opt_dict(self.opt_dict)
            elif phase < 3:
                print('Starting training phase 3')
                phase = 3
                # get a new cache folder
                self.opt_dict['cache_folder'] = split(self.opt_dict['cache_folder'])[0]
                self.opt_dict = self.parse_opt_dict(self.opt_dict)

            sum_loss = None
            loop = tqdm(self.train_data_loader)
            for n, batch in enumerate(loop):
                loss = self.train_step(batch, phase=phase)
                # print(" Epoch: {}, Current loss: {}".format(epoch, loss))
                if sum_loss is None:
                    sum_loss = Counter(loss)
                else:
                    sum_loss += Counter(loss)
                l_str = 'Ep: {}'.format(epoch)
                for l in sum_loss:
                    l_str += ', {}: {:0.5f}'.format(l, sum_loss[l] / (1 + n))
                loop.set_description(l_str)

            for l in sum_loss:
                self.writer.add_scalar(l, sum_loss[l] / len(self.train_data_loader), epoch)

    def iteration_step(self, batch, instance_params, weight_dict={}):
        """
        Computes losses for a single step of optimization.
        Entries in loss/weight dict should have the following entries (always edit loss_keys to modify loss terms):
        corr, templ, s2m, m2s, pose_pr, shape_pr
        """
        loss_keys = ['corr', 'templ', 's2m', 'm2s', 'pose_pr', 'shape_pr']
        for k in loss_keys:
            if k not in weight_dict.keys():
                weight_dict[k] = 1.

        device = self.device
        loss = {}
        scan = batch.get('scan').to(device)
        batch_sz = scan.shape[0]
        # predict initial correspondences
        out = self.model(scan)
        # _, part_label = torch.max(out['part_labels'].data, 1)

        poses, betas, trans = instance_params['pose'], instance_params['betas'], instance_params['trans']
        if 'corr' in instance_params:
            corr = instance_params['corr']
        else:
            corr = out['correspondences'].permute(0, 2, 1)

        # Offset optimization should be implemented here
        if 'offsets' in instance_params:
            offsets = instance_params['offsets']
        else:
            offsets = None

        # Filter out bad correspondences
        # mask = self.filter_correspondences(corr, part_label)

        if self.train_supervised:
            gt_part_label = batch.get('part_labels').to(device).long()
            correspondences = batch.get('correspondences').to(device)
            loss['part_labels'] = F.cross_entropy(out['part_labels'], gt_part_label.squeeze(-1))
            loss['correspondences'] = F.mse_loss(corr, correspondences)

        # get posed smpl points
        template_points = torch.cat([self.template_points] * batch_sz, axis=0).to(device)

        posed_smpl = self.vsmpl(template_points, poses, betas, trans)

        # get posed scan corresponding points
        # import ipdb
        # ipdb.set_trace()
        posed_scan_correspondences = self.vsmpl(corr, poses, betas, trans)

        # correspondence loss
        # loss['corr'] = F.l1_loss(scan * mask, posed_scan_correspondences * mask) * weight_dict['corr']
        loss['corr'] = F.l1_loss(scan, posed_scan_correspondences) * weight_dict['corr']

        # bring scan correspondences in R^3 closer to ref_smpl surface
        ''' Experiment to see if this should be bi-directional or not '''
        loss['templ'] = chamfer_distance(corr, template_points) * weight_dict['templ']

        # chamfer loss
        loss['s2m'] = chamfer_distance(scan, posed_smpl, w2=0) * weight_dict['s2m']
        loss['m2s'] = chamfer_distance(scan, posed_smpl, w1=0) * weight_dict['m2s']

        # pose prior
        loss['pose_pr'] = self.pose_prior(poses).mean() * weight_dict['pose_pr']

        # shape prior
        loss['shape_pr'] = (betas ** 2).mean() * weight_dict['shape_pr']

        return loss

    def train_step(self, batch, phase):
        """
        Each training step consists of multiple iterations of optimization
        """
        # from batch take out instance specific parameters
        pose = batch.get('pose').to(self.device).requires_grad_(True)
        betas = batch.get('betas').to(self.device).requires_grad_(True)
        trans = batch.get('trans').to(self.device).requires_grad_(True)
        instance_params = {'pose': pose, 'betas': betas, 'trans': trans}

        # initialize optimizer for instance specific SMPL params
        smpl_optimizer = self.init_optimizer(self.optimizer_type, instance_params.values(),
                                             learning_rate=0.001 if phase == 3 else 0.002)

        # We only train the regression layers
        self.model.train()
        self.model.set_abstractions.train(False)
        self.model.feature_propagators.train(False)

        self.optimizer.zero_grad()

        # get optimization weights
        wts = self.get_optimization_weights(phase)
        for it in range(self.opt_dict['iter_per_step'][phase]):
            smpl_optimizer.zero_grad()

            loss_ = self.iteration_step(batch, instance_params, weight_dict=wts)
            loss = self.sum_dict(loss_)

            # back propagate
            loss.backward()
            smpl_optimizer.step()

        # Optimize network once per training step
        if phase == 3:
            self.optimizer.step()

        # Save updated instance specific parameters to cache folder
        paths = batch.get('name')
        for n, p in enumerate(paths):
            if not exists(join(p, split(self.opt_dict['cache_folder'])[0])):
                os.makedirs(join(p, split(self.opt_dict['cache_folder'])[0]))

            with open(join(p, self.opt_dict['cache_folder'] + '.pkl'), 'wb') as f:
                pkl.dump({'pose': pose[n].detach().cpu().numpy(),
                          'betas': betas[n].detach().cpu().numpy(),
                          'trans': trans[n].detach().cpu().numpy()},
                         f, protocol=2)
            # print('Saved cache, ', join(p, self.opt_dict['cache_folder'] + '.pkl'))

        return {k: loss_[k].item() for k in loss_}

    def fit_test_sample(self, save_name, num_saves=None):
        from os.path import join, exists
        import os

        epoch = self.load_checkpoint(number=self.checkpoint_number)
        print('Testing with epoch {}'.format(epoch))
        if not exists(join(self.exp_path, save_name + '_ep_{}'.format(epoch))):
            os.makedirs(join(self.exp_path, save_name + '_ep_{}'.format(epoch)))

        self.model.train(False)

        count = 0
        for batch in tqdm(self.val_data_loader):
            names = batch.get('name')
            vcs = batch.get('scan_vc').numpy()
            pose = batch.get('pose').to(self.device).requires_grad_(True)
            betas = batch.get('betas').to(self.device).requires_grad_(True)
            trans = batch.get('trans').to(self.device).requires_grad_(True)

            # predict initial correspondences for saving
            out = self.model(batch.get('scan').to(self.device))
            corr_init = out['correspondences'].permute(0, 2, 1).detach()
            _, part_label = torch.max(out['part_labels'].data, 1)
            corr = corr_init.clone().requires_grad_(True)

            instance_params = {'pose': pose, 'betas': betas, 'trans': trans}

            # initialize optimizer for instance specific SMPL params
            smpl_optimizer = optim.Adam(instance_params.values(), lr=0.02)
            instance_params['corr'] = corr
            corr_optimizer = optim.Adam([corr], lr=0.02)

            for it in range(self.opt_dict['iter_per_step'][1] + self.opt_dict['iter_per_step'][2]):
                smpl_optimizer.zero_grad()
                corr_optimizer.zero_grad()

                if it == 0:
                    phase = 1
                    wts = self.get_optimization_weights(phase=1)
                    print('Optimizing phase 1')
                elif it == self.opt_dict['iter_per_step'][1]:
                    phase = 2
                    wts = self.get_optimization_weights(phase=2)
                    print('Optimizing phase 2')

                loss_ = self.iteration_step(batch, instance_params, weight_dict=wts)
                loss = self.sum_dict(loss_)

                if it % 50 == 0:
                    l_str = 'Iter: {}'.format(it)
                    for l in loss_:
                        l_str += ', {}: {:0.5f}'.format(l, loss_[l].item())
                    print(l_str)

                # back propagate
                loss.backward()
                if phase == 1:
                    smpl_optimizer.step()
                elif phase == 2:
                    smpl_optimizer.step()
                    corr_optimizer.step()

                if it % 300 == 0:
                    pose_ = pose.detach().cpu().numpy()
                    betas_ = betas.detach().cpu().numpy()
                    trans_ = trans.detach().cpu().numpy()
                    corr_ = corr.detach().cpu().numpy()

                    self.save_output(names, pose_, betas_, trans_, corr_, vcs, save_name, epoch, it)

            count += len(names)

            if (num_saves is not None) and (count >= num_saves):
                break

    def save_output(self, names, pose_, betas_, trans_, corr_, vcs, save_name, epoch, it):
        from psbody.mesh import Mesh
        from lib.smpl_paths import SmplPaths

        sp = SmplPaths(gender='male')
        smpl = sp.get_smpl()
        for nam, p, b, t, c, vc in zip(names, pose_, betas_, trans_, corr_, vcs):
            name = split(nam)[1]
            smpl.pose[:] = p
            smpl.betas[:10] = b
            smpl.trans[:] = t

            # save registration
            Mesh(smpl.r, smpl.f).write_ply(
                join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_{}_reg.ply'.format(it)))

            # save raw correspondences
            temp = Mesh(c, [])
            temp.set_vertex_colors(vc)
            temp.write_ply(
                join(self.exp_path, save_name + '_ep_{}'.format(epoch), name + '_{}_craw.ply'.format(it)))

            # save SMPL params
            with open(join(self.exp_path, save_name + '_ep_{}'.format(epoch),
                           name + '_{}_reg.pkl'.format(it)), 'wb') as f:
                pkl.dump({'pose': p, 'betas': b, 'trans': t}, f)