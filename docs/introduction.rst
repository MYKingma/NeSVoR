Introduction
============

Methods
-------

NeSVoR is a deep learning package for solving slice-to-volume reconstruction problems 
(i.e., reconstructing a 3D isotropic high-resolution volume from a set of motion-corrupted low-resolution slices) 
with application to fetal/neonatal MRI, which provides

#. Motion correction by mapping 2D slices to a 3D canonical space using `Slice-to-Volume Registration Transformers (SVoRT) <https://link.springer.com/chapter/10.1007/978-3-031-16446-0_1>`_.
#. Volumetric reconstruction of multiple 2D slices using implicit neural representation `(NeSVoR) <https://ieeexplore.ieee.org/document/10015091>`_.

.. raw:: html

    <p align="center">
        <img src="_static/images/SVoRT_network.png" align="center" width="600">
    </p>
    <p align="center">
        Figure 1. SVoRT: an iterative Transformer for slice-to-volume registration. (a) The k-th iteration of SVoRT. (b) The detailed network architecture of the SVT module.
    <p align="center">

    <p align="center">
        <img src="_static/images/NeSVoR.png" align="center" width="900">
    </p>
    <p align="center">
        Figure 2. NeSVoR: A) The forward imaging model in NeSVoR. B) The architecture of the implicit neural network in NeSVoR.
    <p align="center">


Pipeline
--------

To make our reconstruction tools more handy, we incorporate several preprocessing and downstream analysis tools in this package.
The next figure shows our overall reconstruction pipeline.

.. raw:: html

    <p align="center">
        <img src="_static/images/pipeline.png" align="center" width="900">
    </p>
    <p align="center">
        Figure 3. The reconstruction pipeline.
    <p align="center">