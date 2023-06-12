Preprocessing
-------------

Data preprocessing has a significant impact on the quality of the reconstructed volume in slice-to-volume reconstruciton.
The NeSVoR toolkit provides the following preprocessing steps:

.. raw:: html

    <p align="center">
        <img src="../_static/images/preprocessing.png" align="center" width="600">
    </p>
    <p align="center">
        Preprocessing in NeSVoR toolkit
    <p align="center">

The preprocessing can be performed by using a standalone command 
(i.e., `segment-stack <../commands/segment-stack.html>`__, `correct-bias-field <../commands/correct-bias-field.html>`__, 
and `assess <../commands/assess.html>`__). They can also be integrated into the reconstruction pipeline (i.e., 
`reconstruct <../commands/reconstruct.html>`__ and `svr <../commands/svr.html>`__) by setting the corresponding flags.

Brain masking
^^^^^^^^^^^^^

We integrate a deep learning based fetal brain segmentation model (`MONAIfbs <https://github.com/gift-surg/MONAIfbs>`_) 
into our pipeline to extract the fetal brain ROI from each input image.
The `segment-stack <../commands/segment-stack.html>`__ command generates brain mask for each input stack as follows.

.. code-block:: nesvorcommand

    nesvor segment-stack \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --output-stack-masks mask-1.nii.gz ... mask-N.nii.gz \

You may also perform brain segmentation in the `reconstruct <../commands/reconstruct.html>`__ command by setting 
`--segmentation <../commands/reconstruct.html#segmentation>`__.

Bias field correction
^^^^^^^^^^^^^^^^^^^^^

We also provide a wrapper of 
`the N4 algorithm in SimpleITK <https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html>`_ 
for bias field correction. 
The `correct-bias-field <../commands/correct-bias-field.html>`__ 
command correct the bias field in each input stack and output the corrected stacks.

.. code-block:: nesvorcommand

    nesvor correct-bias-field \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --output-corrected-stacks corrected-stack-1.nii.gz ... corrected-stack-N.nii.gz

You may perform bias field correction in the `reconstruct <../commands/reconstruct.html>`__ 
command by setting `--bias-field-correction <../commands/reconstruct.html#bias-field-correction>`__.

Stack quality assessment
^^^^^^^^^^^^^^^^^^^^^^^^

The `assess <../commands/assess.html>`__ command evalutes the image quality / motion of input stacks. 
This information can be used to find a template stack with the best quality or filter out low-quality data. 
An example is as follows.

.. code-block:: nesvorcommand

    nesvor assess \
        --input-stacks stack-1.nii.gz ... stack-N.nii.gz \
        --stack-masks mask-1.nii.gz ... mask-N.nii.gz \
        --metric <metric> \
        --output-json result.json 

The provided metrics are listed `here <../commands/assess.html#metric>`__.
