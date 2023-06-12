Sampling
--------

NeSVoR learns an implicit representation of the underlying 3D volume. Therefore, after training a NeSVoR model, we can 
sample 3D volumes and 2D slices from it. (Use `--output-moel <../commands/reconstruct.html#output-model>`__ 
in `reconstruct <../commands/reconstruct.html>`__ command to specify the file to save 
the model.)

Sample volume
^^^^^^^^^^^^^

Upon training a NeSVoR model with the `reconstruct <../commands/reconstruct.html>`__ command, 
you can sample a volume at arbitrary resolutions with the `sample-volume <../commands/sample-volume.html>`__ command.

.. code-block:: nesvorcommand

    nesvor sample-volume \
        --output-volume volume.nii.gz \
        --input-model model.pt \
        --output-resolution 0.5

Sample slices
^^^^^^^^^^^^^

You may sample slices from the model using the `sample-slices <../commands/sample-slices.html>`__ command. 
For each slice in ``<path-to-slices-folder>``, 
the command simulates a slice from the NeSVoR model at the corresponding slice location.

.. code-block:: nesvorcommand

    nesvor sample-slices \
        --input-slices <path-to-slices-folder> \
        --input-model model.pt \
        --simulated-slices <path-to-save-simulated-slices>

For example, you can use `sample-slices <../commands/sample-slices.html>`__ to simulate slices at the motion-corrected 
locations and evaluate the reconstruction results by comparing the input slices and the simulated slices. 
