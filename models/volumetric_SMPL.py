"""
This code blows up the canonical SMPL function to the 3D volume.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""

import os
from os.path import join, split, exists
import torch
from torch import nn
import torch.nn.functional as F
import pickle as pkl
import sys
from smplpytorch.smplpytorch.pytorch.tensutils import (th_posemap_axisang, th_with_zeros, th_pack, make_list, subtract_flat_id)
from lib.smpl_layer import SMPL_Layer

def correspondence_to_smpl_function(points, grid, d_grid=None):
    grid = grid.permute(0, 3, 2, 1)
    sz = points.shape
    points_ = points.reshape(1, -1, 3).unsqueeze(1).unsqueeze(1)
    feats = F.grid_sample(grid.unsqueeze(0), points_)
    feats = feats.squeeze(2).squeeze(2).view(-1, sz[0], sz[1]).permute(1, 2, 0)
    return feats

class VolumetricSMPL(nn.Module):
    def __init__(self, folder, device, gender='male'):
        super(VolumetricSMPL, self).__init__()

        with torch.no_grad():
            # Load transformation
            with open(join(folder, 'scale_center.pkl'), 'rb') as f:
                self.scale, self.center = pkl.load(f, encoding='latin-1')
                self.scale = torch.tensor(self.scale.astype('float32'), requires_grad=False).to(device)
                self.center = torch.tensor(self.center.astype('float32'), requires_grad=False).to(device)

            # Load closest_point
            with open(join(folder, 'closest_point.pkl'), 'rb') as f:
                # Shape: 3(x, y, z) x res x res x res
                closest_point = pkl.load(f, encoding='latin-1').astype('float32')
                self.closest_point = torch.tensor(closest_point,
                                                  requires_grad=False).permute(3, 0, 1, 2).to(device)
                res = self.closest_point.shape[-1]

            # Load shapedirs
            with open(join(folder, 'shapedirs.pkl'), 'rb') as f:
                # Shape: 10 x 3(x, y, z) x res x res x res
                shapedirs = pkl.load(f, encoding='latin-1')[..., :10].astype('float32')  # keep only dim shape dims
                self.shapedirs = torch.tensor(shapedirs.reshape(res, res, res, -1),
                                              requires_grad=False).permute(3, 0, 1, 2).to(device)

            # Load posedirs
            with open(join(folder, 'posedirs.pkl'), 'rb') as f:
                # Shape: 207 x 3 x res x res x res
                posedirs = pkl.load(f, encoding='latin-1').astype('float32')
                self.posedirs = torch.tensor(posedirs.reshape(res, res, res, -1),
                                             requires_grad=False).permute(3, 0, 1, 2).to(device)

            # Load skinning_weights
            with open(join(folder, 'skinning_weights.pkl'), 'rb') as f:
                skinning_weights = pkl.load(f, encoding='latin-1').astype('float32')
                self.skinning_weights = torch.tensor(skinning_weights,
                                                     requires_grad=False).permute(3, 0, 1, 2).to(device)

        self.grid_fn = correspondence_to_smpl_function
        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                               model_root='smplpytorch/smplpytorch/native/models').to(device)

    def transform_points(self, points):
        return points * self.scale + self.center

    def compute_smpl_skeleton(self, th_betas, th_pose_axisang):
        batch_size = th_pose_axisang.shape[0]
        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(th_pose_axisang)
        # Take out the first rotmat (global rotation)
        root_rot = th_pose_rotmat[:, :9].view(batch_size, 3, 3)
        # Take out the remaining rotmats (23 joints)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        th_v = self.smpl.th_v_template + torch.matmul(
            self.smpl.th_shapedirs[..., :10], th_betas.transpose(1, 0)).permute(2, 0, 1)
        th_j = torch.matmul(self.smpl.th_J_regressor, th_v)

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(self.smpl.num_joints - 1):
            i_val = int(i + 1)
            joint_rot = th_pose_rotmat[:, (i_val - 1) * 9:i_val *
                                                          9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self.smpl.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))

        th_results2 = torch.zeros((batch_size, 4, 4, self.smpl.num_joints),
                                  dtype=root_j.dtype,
                                  device=root_j.device)

        for i in range(self.smpl.num_joints):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        return th_results2

    def compute_transformed_points(self, points, pose, beta, trans):
        """
        Takes points in R^3 and first applies relevant pose and shape blend shapes.
        Then performs skinning.
        """
        batch_size = pose.shape[0]
        posedirs = self.grid_fn(points, self.posedirs)
        shapedirs = self.grid_fn(points, self.shapedirs)
        skinning_weights = self.grid_fn(points, self.skinning_weights)
        closest_point = self.grid_fn(points, self.closest_point)

        # reshape back
        shapedirs = shapedirs.view(batch_size, -1, 3, 10)
        posedirs = posedirs.view(batch_size, -1, 3, 207)
        skinning_weights = skinning_weights.view(batch_size, -1, 24)

        # Convert axis-angle representation to rotation matrix rep.
        th_pose_rotmat = th_posemap_axisang(pose)
        th_pose_rotmat = th_pose_rotmat[:, 9:]
        th_pose_map = subtract_flat_id(th_pose_rotmat)

        p_v_shaped = closest_point + (shapedirs * beta.unsqueeze(1).unsqueeze(1)).sum(-1)

        p_v_posed = p_v_shaped + (posedirs * th_pose_map.unsqueeze(1).unsqueeze(1)).sum(-1)

        th_results2 = self.compute_smpl_skeleton(beta, pose)

        # Skinning
        p_T = torch.bmm(th_results2.view(-1, 16, 24), skinning_weights.transpose(2, 1)).view(batch_size, 4, 4, -1)
        p_rest_shape_h = torch.cat([
            p_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, p_v_posed.shape[1]),
                       dtype=p_T.dtype,
                       device=p_T.device),
        ], 1)

        p_verts = (p_T * p_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        p_verts = p_verts[:, :, :3]

        p_verts = p_verts + trans.unsqueeze(1)
        return p_verts

    def forward(self, points, pose, beta, trans, offsets=None):
        """
        :param points: correspondences in R^3. Must be normalized.
        :param pose, shape, trans: SMPL params
        """
        # normalize the correspondences
        points_ = self.transform_points(points)

        # Compute forward SMPL function on correspondences
        p_tr = self.compute_transformed_points(points_, pose, beta, trans)

        return p_tr


def test():
    name = join('assets/rp_eric_rigged_005_zup_a_smpl.obj')

    # Load SMPL params
    dat = pkl.load(open(name.replace('_smpl.obj', '_param.pkl'), 'rb'), encoding='latin-1')
    vsmpl = VolumetricSMPL('assets/volumetric_smpl_function_64',
                           'cuda', 'male')

    '''Sanity check: transform SMPL vertices using SMPL and VolumetricSMPL'''
    corr = init_SMPL('male', dat)
    corr.pose[:] = 0
    corr.trans[:] = 0
    corr.betas[:] = 0
    p = torch.tensor([corr.r.astype('float32')] * 2).to('cuda')
    b = torch.tensor([dat['betas'][:10].astype('float32')] * 2).to('cuda')
    po = torch.tensor([dat['pose'].astype('float32')] * 2).to('cuda')
    t = torch.tensor([dat['trans'].astype('float32')] * 2).to('cuda')
    ptr = vsmpl(p, po, b, t)

    Mesh(ptr[0].cpu().numpy(), []).write_ply('assets/test.ply')
    corr = init_SMPL('male', dat)
    Mesh(corr.r, corr.f).write_ply('assets/org.ply')

    print('Done')


if __name__ == "__main__":
    from psbody.mesh import Mesh
    import numpy as np
    from lib.smpl_helpers import init_SMPL

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    test()
    print('Done')