Get transformations of motion-corrected slices
==============================================

The commands `reconstruct <../commands/reconstruct.html#output-slices>`__, 
`svr <../commands/svr.html#output-slices>`__, and `register <../commands/register.html#output-slices>`__ 
allow us to save the motion-corrected slices to a folder by 
setting parameter ``--output-slices`` as follows

.. code-block:: nesvorcommand

    nesvor reconstruct \
        ...... \
        --output-slices folder-to-save-slices \
        ......

The slices and the corresponding masks would be saved to separate ``.nii`` files in the folder.

.. code-block:: console

    $ ls folder-to-save-slices -rt | head
    0.nii.gz
    mask_0.nii.gz
    1.nii.gz
    mask_1.nii.gz
    2.nii.gz
    mask_2.nii.gz
    3.nii.gz
    mask_3.nii.gz
    4.nii.gz
    mask_4.nii.gz


The transformation of each slice after motion correction can be obtained from the affine matrix in the ``.nii`` file. 

.. code-block:: pycon

    >>> import nibabel as nib
    >>> nib_image = nib.load('folder-to-save-slices/0.nii.gz')
    >>> nib_image.affine
    array([[ 1.08141804e+00, -7.78946728e-02, -2.63444871e-01, -1.09909004e+02],
       [-9.95342284e-02,  4.51498836e-01, -1.81252944e+00, -3.23390656e+01],
       [ 1.30065709e-01,  9.93161798e-01,  8.03327262e-01, -1.97971268e+02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

