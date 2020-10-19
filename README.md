# LoopReg
Repo for **"LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20 (Oral)"**

## Prerequisites
1. Cuda 10.0
2. Cudnn 7.6.5
3. Kaolin (https://github.com/NVIDIAGameWorks/kaolin) - for SMPL registration
4. MPI mesh library (https://github.com/MPI-IS/mesh)
5. Trimesh
6. Python 3.7.6
7. Tensorboard 1.15
8. Pytorch 1.4
9. SMPL pytorch from https://github.com/gulvarol/smplpytorch. I have included these files (with required modifications) in this repo.
10. Download SMPL from https://smpl.is.tue.mpg.de/

## Download pre-trained models
1. Download LoopReg weights:
2. `mkdir <LoopReg directory>/experiments`
3. Put the downloaded weights in `<LoopReg directory>/experiments/`

## Test LoopReg
0. Spread SMPL from mesh surface to R^3.
`python spread_SMPL_function.py`
1. Make data split. Adjust paths in the scripts and run `make_data_split.py`.
2. Test LoopReg.
`python train_PartSpecificNet.py 1 -mode val -save_name corr -batch_size 16 -split_file assets/data_split_01_unsupervised.pkl`


For training/ testing on dataset, you'd need the following directory structure if you'd like to use our dataloaders:

[DATASETS]\
-[dataset]\
--[subject_01]\
---[scan.obj]\

## Train LoopReg
0. Spread SMPL from mesh surface to R^3.
`python spread_SMPL_function.py`
1. Make data split. Adjust paths in the script and run `make_data_split.py`. Make desired split for supervised and unsupervised training.
2. Warm start correspondence predictor with small amount of supervised data.
`python warup_PartSpecificNet.py -batch_size 16 -split_file assets/data_split_01_unsupervised.pkl`
3. Jointly optimise the correspondence network and SMPL parameters using self-supervised training.
`python train_PartSpecificNet.py 1 -batch_size 16 -cache_suffix cache_1 -split_file assets/data_split_01_supervised.pkl`


Cite us:
```
@inproceedings{bhatnagar2020loopreg,
    title = {LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration},
    author = {Bhatnagar, Bharat Lal and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {Neural Information Processing Systems (NeurIPS)},
    month = {December},
    year = {2020},
}
```

## License

Copyright (c) 2020 Bharat Lal Bhatnagar, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration** paper in documents and papers that report on research using this Software.