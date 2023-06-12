Reconstruction and motion correction
====================================

NeSVoR reconstruction
-----------------------

Reconstruct from mutiple stacks of slices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `reconstruct <../commands/reconstruct.html>`__ command can be used to reconstruct a 3D volume 
from ``N`` stacks of 2D slices (in NIfTI format, i.e. ``.nii`` or ``.nii.gz``). 
A basic usage of `reconstruct <../commands/reconstruct.html>`__ is as follows, 
where ``mask-i.nii.gz`` is the ROI mask of the ``i``-th input stack.

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --output-volume volume.nii.gz

A more elaborate example could be 

.. code-block:: nesvorcommand

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

Run ``nesvor reconstruct --h`` to see the meaning of each parameter.

Given multiple stacks at inputs, 
`reconstruct <../commands/reconstruct.html>`__ first corrects the motion in the input stacks using SVoRT 
(the same as what `register <../commands/register.html>`__ does), 
and then trains a NeSVoR model that implicitly represent the underlying 3D volume, 
from which a discretized volume (i.e., a 3D tensor) can be sampled.

Reconstruct from motion-corrected slices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`reconstruct <../commands/reconstruct.html>`__ can also take a folder of motion-corrected slices as inputs. 

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-slices <path-to-slices-folder> \
        --output-volume volume.nii.gz

This enables the separation of registration and reconstruction. 
That is, you may first run `register <../commands/register.html>`__ to perform motion correction, 
and then use `reconstruct <../commands/reconstruct.html>`__ 
to reconstruct a volume from a set of motion-corrected slices.

Deformable NeSVoR
^^^^^^^^^^^^^^^^^

NeSVoR can now reconstruct data with deformable (non-rigid) motion! 
To enable deformable motion, use the flag `--deformable <../commands/reconstruct.html#deformable>`__. 

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --deformable \
        .....

This feature is still experimental.

SVoRT motion correction
--------------------------------

`register <../commands/register.html>`__ takes mutiple stacks of slices as inputs, performs motion correction, 
and then saves the motion-corrected slices to a folder.

.. code-block:: nesvorcommand

    nesvor register \
    --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
    --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
    --registration <method> \
    --output-slices <path-to-save-output-slices>

The list of supported methods in `register <../commands/register.html>`__ 
can be found `here <commands/register.html#registration>`__.


Slice-to-volume registration/reconstruction
----------------------------------------------

`svr <../commands/svr.html>`__ implements a classical slice-to-volume registration/reconstruction method combined with SVoRT
motion correction. THe usage of `svr <../commands/svr.html>`__ is similar to `reconstruct <../commands/reconstruct.html>`__.
`svr <../commands/svr.html>`__ currently only supports rigid motion.

.. code-block:: nesvorcommand

    nesvor reconstruct \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --output-volume volume.nii.gz