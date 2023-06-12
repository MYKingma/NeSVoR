# NeSVoR: **Ne**ural **S**lice-to-**Vo**lume **R**econstruction

<p align="left">
  <a href="https://github.com/daviddmc/NeSVoR/releases" alt="releases">
    <img src="https://img.shields.io/github/v/release/daviddmc/NeSVoR?display_name=tag" />
  </a>
  <a href="https://hub.docker.com/r/junshenxu/nesvor/tags" alt="docker">
    <img src="https://img.shields.io/docker/v/junshenxu/nesvor?label=docker%20image%20version" />
  </a>
  <a href='https://nesvor.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/nesvor/badge/?version=latest' alt='Documentation Status' />
  </a>
</p>

NeSVoR is a package for GPU-accelerated slice-to-volume reconstruction (both rigid and deformable).

This package is the accumulation of the following works:

\[1\] SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI 
([Springer](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1) | [Arxiv](https://arxiv.org/abs/2206.10802))

\[2\] NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI 
([IEEE](https://ieeexplore.ieee.org/document/10015091) | [TechRxiv](https://www.techrxiv.org/articles/preprint/NeSVoR_Implicit_Neural_Representation_for_Slice-to-Volume_Reconstruction_in_MRI/21398868/1))

<p align="center">
   <img src="./docs/_static/images/recon.gif" align="center" width="600">
</p>
<p align="center"><p align="center">

<!-- toc -->
- [Overview](#overview)
  - [Methods](#methods)
  - [Pipeline](#pipeline)
- [Documentation](#documentation)
- [Installation](#installation)
  - [Docker Image](#docker-image)
  - [From Source](#from-source)
- [Quick Start](#quick-start)
  - [Fetal Brain Reconstruction](#fetal-brain-reconstruction)
  - [Neonatal Brain Reconstruction](#neonatal-brain-reconstruction)
  - [Fetal Body/Uterus Reconstruction](#fetal-bodyuterus-reconstruction)
- [Usage](#usage)
  - [Reconstruction ](#reconstruction)
  - [Registration (Motion Correction)](#registration-motion-correction)
  - [SVR](#svr)
  - [Sampling](#sampling)
    - [Sample Volume](#sample-volume)
    - [Sample Slices](#sample-slices)
  - [Preprocessing](#preprocessing)
    - [Brain Masking](#brain-masking)
    - [Bias Field Correction](#bias-field-correction)
    - [Stack Quality Assessment](#stack-quality-assessment) <!--  - [3D Brain Segmentation](#3d-brain-segmentation) -->
- [Cite Our Work](#cite-our-work)
- [Resources](#resources)

<!-- tocstop -->

## Overview

### Methods

NeSVoR is a deep learning package for solving slice-to-volume reconstruction problems (i.e., reconstructing a 3D isotropic high-resolution volume from a set of motion-corrupted low-resolution slices) with application to fetal/neonatal MRI, which provides
- Motion correction by mapping 2D slices to a 3D canonical space using [Slice-to-Volume Registration Transformers (SVoRT)](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1).
- Volumetric reconstruction of multiple 2D slices using implicit neural representation ([NeSVoR](https://www.techrxiv.org/articles/preprint/NeSVoR_Implicit_Neural_Representation_for_Slice-to-Volume_Reconstruction_in_MRI/21398868/1)).

<!--### Slice-to-Volume Registration Transformers (SVoRT)-->

<p align="center">
   <img src="./docs/_static/images/SVoRT_network.png" align="center" width="600">
</p>
<p align="center">Figure 1. SVoRT: an iterative Transformer for slice-to-volume registration. (a) The k-th iteration of SVoRT. (b) The detailed network architecture of the SVT module.<p align="center">

<!--### Neural Slice-to-Volume Reconstruction (NeSVoR)-->

<p align="center">
   <img src="./docs/_static/images/NeSVoR.png" align="center" width="900">
</p>
<p align="center">Figure 2. NeSVoR: A) The forward imaging model in NeSVoR. B) The architecture of the implicit neural network in NeSVoR.<p align="center">

### Pipeline

To make our reconstruction tools more handy, we incorporate several preprocessing and downstream analysis tools in this package.
The next figure shows our overall reconstruction pipeline.

<p align="center">
   <img src="./docs/_static/images/pipeline.png" align="center" width="900">
</p>
<p align="center">Figure 3. The reconstruction pipeline.<p align="center">

## Documentation

The full documentation is available at [Read the Docs](https://nesvor.readthedocs.io/).

## Installation

### Docker Image

We recommend to use our docker image to run `nesvor`.

#### Install Docker and NVIDIA Container Toolkit

You may follow this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to install Docker and NVIDIA Container Toolkit

#### Download NeSVoR Image

```
docker pull junshenxu/nesvor
```
Note: our latest image was built with CUDA 11.7.

#### Run NeSVoR with Docker

You may run a container in an iterative way.

```
docker run -it --gpus all junshenxu/nesvor
nesvor -h
```

You may also run the `nesvor` command directly as follows.

```
docker run --rm --gpus all \
    -v <path-to-inputs>:/incoming:ro -v <path-to-outputs>:/outgoing:rw \
    junshenxu/nesvor \
    nesvor reconstruct \
    --input-stacks /incoming/stack-1.nii.gz ... /incoming/stack-N.nii.gz \
    --thicknesses <thick-1> ... <thick-N> \
    --output-volume /outgoing/volume.nii.gz
```

### From Source

Check out our [documentation](https://nesvor.readthedocs.io/en/latest/installation.html#from-source) if you want to install NeSVoR from source.

## Quick Start

### Fetal Brain Reconstruction

This example reconstruct a 3D fetal brain from mutiple stacks of 2D images in the following steps:

1. Segment fetal brain from each image using a CNN (`--segmentation`).
2. Correct bias field in each stack using the N4 algorithm (`--bias-field-correction`).
3. Register slices using SVoRT (`--registration svort`).
4. Reconstruct a 3D volume using NeSVoR.

```
nesvor reconstruct \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --thicknesses <thick-1> ... <thick-N> \
    --output-volume volume.nii.gz \
    --output-resolution 0.8 \
    --registration svort \
    --segmentation \
    --bias-field-correction
```

### Neonatal Brain Reconstruction

This example reconstruct a 3D neonatal brain from mutiple stacks of 2D images in the following steps:

1. Removal background (air) with Otsu thresholding (`--otsu-thresholding`).
2. Correct bias field in each stack using the N4 algorithm (`--bias-field-correction`).
3. Register slices using SVoRT (`--registration svort`).
4. Reconstruct a 3D volume using NeSVoR.

```
nesvor reconstruct \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --thicknesses <thick-1> ... <thick-N> \
    --output-volume volume.nii.gz \
    --output-resolution 0.8 \
    --registration svort \
    --otsu-thresholding \
    --bias-field-correction
```

### Fetal Body/Uterus Reconstruction

This is an example for deformable NeSVoR which consists of the following steps:

1. Create an ROI based on the intersection of all input stacks (`--stacks-intersection`).
3. Perform stack-to-stack registration (`--registration stack`).
3. Reconstruct a 3D volume using Deformable NeSVoR (`--deformable`).

```
nesvor reconstruct \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --thicknesses <thick-1> ... <thick-N> \
    --output-volume volume.nii.gz \
    --output-resolution 1.0 \
    --stacks-intersection \
    --registration stack \
    --deformable \
    --weight-transformation 1 \
    --weight-deform 0.1 \
    --weight-image 0.1 \
    --single-precision \
    --log2-hashmap-size 22 \
    --batch-size 8192
```

## Usage

This section provides the basic usage of the commands in NeSVoR.
Please refer to our [document](https://nesvor.readthedocs.io/en/latest/cmd.html) for details.
NeSVoR currently supports the following commands:

- [`nesvor reconstruct`](#reconstruction): reconstruct a 3D volume (i.e., train a NeSVoR model) from either multiple stacks of slices (NIfTI) or a set of motion-corrected slices (the output of `register`). It can also perform multiple preprocessing steps, including brain segmentation, bias field correction, and registration.
- [`nesvor register`](#registration-motion-correction): register stacks of slices using a pretrained SVoRT model or stack-to-stack registration.
- [`svr`](#svr): a classical slice-to-volume registration/reconstruciton method.
- [`nesvor sample-volume`](#sample-volume): sample a volume from a trained NeSVoR model.
- [`nesvor sample-slices`](#sample-slices): simulate slices from a trained NeSVoR model.
- [`nesvor segment-stack`](#brain-masking): 2D fetal brain segmentation/masking in input stacks.
- [`nesvor correct-bias-field`](#bias-field-correction): bias field correction using the N4 algorihtm.
- [`nesvor assess`](#stack-quality-assessment): quality and motion assessment of input stacks.
<!-- - [`nesvor segment-volume`](#3d-brain-segmentation): 3D fetal brain segmentation in reconstructed volumes. -->

run `nesvor <command> -h` for a full list of parameters of a command.

### Reconstruction

#### Reconstruct from Mutiple Stacks of Slices

The `reconstruct` command can be used to reconstruct a 3D volume from `N` stacks of 2D slices (in NIfTI format, i.e. `.nii` or `.nii.gz`). 
A basic usage of `reconstruct` is as follows, where `mask-i.nii.gz` is the ROI mask of the i-th input stack.

```
nesvor reconstruct \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --output-volume volume.nii.gz
```

A more elaborate example could be

```
nesvor reconstruct \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --thicknesses <thick-1> ... <thick-N> \
    --output-volume volume.nii.gz \
    --output-resolution 0.8 \
    --output-model model.pt \
    --weight-image 1.0 \
    --image-regularization edge \
    --delta 0.2 \
    --n-iter 5000 \
    --level-scale 1.38 \
    --coarsest-resolution 16.0 \
    --finest-resolution 0.5 \
    --n-levels-bias 0 \
    --n-samples 128
```

Run `nesvor reconstruct --h` to see the meaning of each parameter.

Given multiple stacks at inputs, `reconstruct` first corrects the motion in the input stacks using SVoRT (the same as what `register` did), 
and then trains a NeSVoR model that implicitly represent the underlying 3D volume, from which a discretized volume (i.e., a 3D tensor) can be sampled.

#### Reconstruct from Motion-Corrected Slices

`reconstruct` can also take a folder of motion-corrected slices as inputs. 

```
nesvor reconstruct \
    --input-slices <path-to-slices-folder> \
    --output-volume volume.nii.gz
```
This enables the separation of registration and reconstruction. That is, you may first run `register` to perform motion correction, and then use `reconstruct` to reconstruct a volume from a set of motion-corrected slices.

#### Deformable NeSVoR

NeSVoR can now reconstruct data with deformable (non-rigid) motion! To enable deformable motion, use the flag `--deformable`.

```
nesvor reconstruct \
    --deformable \
    ......
```
This feature is still experimental.

### Registration (Motion Correction)

`register` takes mutiple stacks of slices as inputs, performs motion correction, and then saves the motion-corrected slices to a folder.

```
nesvor register \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --registration <method> \
    --output-slices <path-to-save-output-slices>
```

run `nesvor register -h` to see a full list of supported registration methods.

### SVR

`svr` implements a classical slice-to-volume registration/reconstruction method combined with SVoRT
motion correction. The usage of `svr` is similar to `reconstruct`.
`svr` currently only supports rigid motion.

### Sampling

#### Sample Volume

Upon training a NeSVoR model with the `reconstruct` command, you can sample a volume at arbitrary resolutions with the `sample-volume` command.

```
nesvor sample-volume \
    --output-volume volume.nii.gz \
    --input-model model.pt \
    --output-resolution 0.5
```

#### Sample Slices

You may sample slices from the model using the `sample-slices` command. For each slice in `<path-to-slices-folder>`, the command simulates a slice from the NeSVoR model at the corresponding slice location.

```
nesvor sample-slices \
    --input-slices <path-to-slices-folder> \
    --input-model model.pt \
    --simulated-slices <path-to-save-simulated-slices>
```

For example, after running `reconstruct`, you can use `sample-slices` to simulate slices at the motion-corrected locations and evaluate the reconstruction results by comparing the input slices and the simulated slices. 

### Preprocessing

#### Brain Masking

We integrate a deep learning based fetal brain segmentation model ([MONAIfbs](https://github.com/gift-surg/MONAIfbs)) into our pipeline to extract the fetal brain ROI from each input image. 
Check out their [repo](https://github.com/gift-surg/MONAIfbs) and [paper](https://arxiv.org/abs/2103.13314) for details. 
The `segment-stack` command generates brain mask for each input stack as follows.

```
nesvor segment-stack \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --output-stack-masks mask-1.nii.gz ... mask-N.nii.gz \
```

You may also perform brain segmentation in the `reconstruct` command by setting `--segmentation`.

#### Bias Field Correction

We also provide a wrapper of [the N4 algorithm in SimpleITK](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html) for bias field correction. 
The `correct-bias-field` command correct the bias field in each input stack and output the corrected stacks.

```
nesvor correct-bias-field \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --output-corrected-stacks corrected-stack-1.nii.gz ... corrected-stack-N.nii.gz
```

You may perform bias field correction in the `reconstruct` command by setting `--bias-field-correction`

#### Stack Quality Assessment

The `assess` command evalutes the image quality/motion of input stacks. 
This information can be used to find a template stack with the best quality or filter out low-quality data. 
An example is as follows.

```
nesvor assess \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --metric <metric> \
    --output-json result.json 
```

run `nesvor assess -h` to see a full list of supported metrics.

<!--

### 3D Brain Segmentation

The coherent 3D volume generated by our pipeline can be used for downstream analysis, for example, segmentation or parcellation of 3D brain volume. 
The `segment-volume` command provides a wrapper of the TWAI segmentation algorithm for T2w fetal brain MRI. 
You may find more detials of this method in the authors' [repo](https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation). 
To use this tool, you need to clone their repo and update the path in `config.py` (see the comment in `config.py` for details). 
An exmaple of `segment-volume` is as follows:

```
nesvor segment-volume
    --input-volume reconstructed-volume.nii.gz \
    --output-folder <path-to-save-segmentation>
```

-->

## Cite Our Work

SVoRT
```
@inproceedings{xu2022svort,
  title={SVoRT: Iterative Transformer for Slice-to-Volume Registration in Fetal Brain MRI},
  author={Xu, Junshen and Moyer, Daniel and Grant, P Ellen and Golland, Polina and Iglesias, Juan Eugenio and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={3--13},
  year={2022},
  organization={Springer}
}
```

NeSVoR
```
@article{10015091,
  author={Xu, Junshen and Moyer, Daniel and Gagoski, Borjan and Iglesias, Juan Eugenio and Ellen Grant, P. and Golland, Polina and Adalsteinsson, Elfar},
  journal={IEEE Transactions on Medical Imaging}, 
  title={NeSVoR: Implicit Neural Representation for Slice-to-Volume Reconstruction in MRI}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3236216}
}
```

Fetal IQA
```
@inproceedings{xu2020semi,
  title={Semi-supervised learning for fetal brain MRI quality assessment with ROI consistency},
  author={Xu, Junshen and Lala, Sayeri and Gagoski, Borjan and Abaci Turk, Esra and Grant, P Ellen and Golland, Polina and Adalsteinsson, Elfar},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={386--395},
  year={2020},
  organization={Springer}
}
```

## Resources

This project has been greatly inspired by the following list of fantastic works.

- [gift-surg/NiftyMIC](https://github.com/gift-surg/NiftyMIC)
- [bkainz/fetalReconstruction](https://github.com/bkainz/fetalReconstruction)
- [SVRTK](https://github.com/SVRTK/SVRTK)
- [daviddmc/fetal-IQA](https://github.com/daviddmc/fetal-IQA)
- [FNNDSC/pl-fetal-brain-assessment](https://github.com/FNNDSC/pl-fetal-brain-assessment)
- [LucasFidon/trustworthy-ai-fetal-brain-segmentation](https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation)
- [gift-surg/MONAIfbs](https://github.com/gift-surg/MONAIfbs)