"""
Create a discrete voxel grid and spread SMPL function from surface to the volume.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""
import os
import numpy as np
import pickle as pkl
from psbody.mesh import Mesh
from os.path import exists, split, join
from lib.smpl_paths import SmplPaths


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def create_grid_pts(res=128):
    x_ = np.linspace(-1, 1., res)
    y_ = np.linspace(-1, 1., res)
    z_ = np.linspace(-1, 1., res)

    x, y, z = np.meshgrid(x_, y_, z_)
    pts = np.concatenate([y.reshape(-1, 1), x.reshape(-1, 1), z.reshape(-1, 1)], axis=-1)
    return pts


def process_shapedirs(shapedirs, vert_ids, bary_coords):
    arr = []
    for i in range(3):
        t = barycentric_interpolation(shapedirs[:, i, :][vert_ids], bary_coords)
        arr.append(t[:, np.newaxis, :])
    arr = np.concatenate(arr, axis=1)
    return arr


def transform_points(pts, scale, trans, reverse=False):
    if reverse:
        return (pts - trans)/scale
    return (pts * scale) + trans


def main(res, save_dir):
    if not exists(save_dir):
        os.makedirs(save_dir)

    sp = SmplPaths(gender='male')
    smpl = sp.get_smpl()
    smpl_mesh = Mesh(smpl.r, smpl.f)

    # Bring SMPL mesh to [-1, 1]. Scaling such that height is 1.6m and center is 0
    height = max(smpl_mesh.v.max(axis=0) - smpl_mesh.v.min(axis=0))
    scale = TGT_HEIGHT / height
    smpl_mesh.v *= scale

    center = (smpl_mesh.v.max(axis=0) + smpl_mesh.v.min(axis=0))/2
    center = TGT_CENTER - center
    smpl_mesh.v += center

    '''Save the transformation'''
    if not exists(join(save_dir, 'scale_center.pkl')) or REDO == True:
        with open(join(save_dir, 'scale_center.pkl'), 'wb') as f:
            pkl.dump([scale, center], f, protocol=2)
    else:
        print('scale_center already exists')

    pts = create_grid_pts(res=res)  # shape: res x res x res x 3; range: [-1, 1]
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))

    # Check if interpolation is working as desired
    # assert closest_points == barycentric_interpolation(smpl_mesh.v[vert_ids], bary_coords)

    '''Save closest point'''
    if exists(join(save_dir, 'closest_point.pkl')) and REDO == False:
        print('closest_point already exist')
    else:
        # save original values
        tr_closest_points = transform_points(closest_points, scale, center, reverse=True)
        tr_closest_points = tr_closest_points.reshape(res, res, res, 3)
        with open(join(save_dir, 'closest_point.pkl'), 'wb') as f:
            pkl.dump(tr_closest_points, f, protocol=2)
        print('Saved closest_point', tr_closest_points.shape)

    '''Save distance to closest point'''
    if exists(join(save_dir, 'closest_distance.pkl')) and REDO == False:
        print('closest_distance already exist')
    else:
        # save original values
        closest_distance = ((pts - closest_points)**2).sum(axis=-1).reshape(res, res, res)**0.5 / scale
        with open(join(save_dir, 'closest_distance.pkl'), 'wb') as f:
            pkl.dump(closest_distance, f, protocol=2)
        print('Saved closest_distance', closest_distance.shape)

    '''Interpolate shape dirs'''
    # Due to memory limitation we run interpolation independently for x, y, z
    if exists(join(save_dir, 'shapedirs.pkl')) and REDO == False:
        print('Shapedirs already exist')
    else:
        sdir = process_shapedirs(smpl.shapedirs, vert_ids, bary_coords)
        sdir = sdir.reshape(res, res, res, 3, -1)
        with open(join(save_dir, 'shapedirs.pkl'), 'wb') as f:
            pkl.dump(sdir, f, protocol=2)
        print('Saved shapedirs', sdir.shape)
        del sdir

    '''Interpolate pose dirs'''
    # Due to memory limitation we run interpolation independently for x, y, z
    if exists(join(save_dir, 'posedirs.pkl')) and REDO == False:
        print('Posedirs already exist')
    else:
        pdir = process_shapedirs(smpl.posedirs, vert_ids, bary_coords)
        pdir = pdir.reshape(res, res, res, 3, -1)
        with open(join(save_dir, 'posedirs.pkl'), 'wb') as f:
            pkl.dump(pdir, f, protocol=2)
        print('Saved posedirs', pdir.shape)
        del pdir

    '''Interpolate skinning weights'''
    if exists(join(save_dir, 'skinning_weights.pkl')) and REDO == False:
        print('skinning_weights already exist')
    else:
        skinning_weights = barycentric_interpolation(smpl.weights[vert_ids], bary_coords)
        skinning_weights = skinning_weights.reshape(res, res, res, -1)
        with open(join(save_dir, 'skinning_weights.pkl'), 'wb') as f:
            pkl.dump(np.array(skinning_weights), f, protocol=2)
        print('Saved skinning_weights', skinning_weights.shape)

    print('Done')


REDO = False
TGT_HEIGHT = 1.6
TGT_CENTER = 0.
if __name__ == "__main__":
    res = 64
    main(res, save_dir='assets/volumetric_smpl_function_{}'.format(res))
