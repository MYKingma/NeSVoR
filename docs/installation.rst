Installation
============

Docker image
------------

We recommend to use our docker image to run ``nesvor``.

Install docker and NVIDIA container toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may follow this `guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ to install docker and NVIDIA container toolkit.

Download NeSVoR image
^^^^^^^^^^^^^^^^^^^^^

Pull the latest NeSVoR image from docker hub.

.. code-block:: bash

    docker pull junshenxu/nesvor

**Note**: our latest image was built with CUDA 11.7.

Run NeSVoR with docker
^^^^^^^^^^^^^^^^^^^^^^

You may run a container in an iterative way.

.. code-block:: bash

    docker run -it --gpus all junshenxu/nesvor:v-version-placeholder-
    nesvor -h

You may also run the ``nesvor`` command directly as follows.

.. code-block:: bash

    docker run --rm --gpus all \
        -v <path-to-inputs>:/incoming:ro -v <path-to-outputs>:/outgoing:rw \
        junshenxu/nesvor:v-version-placeholder- \
        nesvor reconstruct \
        --input-stacks /incoming/stack-1.nii.gz ... /incoming/stack-N.nii.gz \
        --thicknesses <thick-1> ... <thick-N> \
        --output-volume /outgoing/volume.nii.gz

From source
------------

Prerequisites
^^^^^^^^^^^^^

If you are installing from source, you will need:

#. An NVIDIA GPU;
#. Python 3.8 or later;
#. GCC/G++ 7.5 or higher;
#. CUDA v10.2 or higher;
#. CMake v3.21 or higher.

See `tiny-cuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_ for more about prerequisites.

Get the NeSVoR source
^^^^^^^^^^^^^^^^^^^^^

Since the master branch might be messy sometimes, it is recommanded to clone a specific release.

.. code-block:: bash

    git clone https://github.com/daviddmc/NeSVoR --branch v-version-placeholder-
    cd NeSVoR


Install dependencies
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -r requirements.txt

Install PyTorch extension of `tiny-cuda-nn <https://github.com/NVlabs/tiny-cuda-nn>`_. 
Make sure the installed CUDA version mismatches the version that was used to compile PyTorch. 
Then, run the following command (see `this <https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension>`_ for more details)

.. code-block:: bash

    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

Other dependencies
^^^^^^^^^^^^^^^^^^

Some functionalities of NeSVoR require extra dependencies.

Fetal brain masking (MONAIfbs)
++++++++++++++++++++++++++++++

To use the MONAIfbs model for fetal brain ROI masking, `MONAI <https://monai.io/>`_ need to be installed.
Check out the orginal `repo <https://github.com/gift-surg/MONAIfbs>`__ for more details.

.. code-block:: bash

   pip install monai>=0.3.0

N4 bias field correction
++++++++++++++++++++++++

To use the N4 algorithm for bias field correction, `SimpleITK <https://simpleitk.readthedocs.io/>`_ need to be installed.

.. code-block:: bash

   pip install SimpleITK

3D IQA
++++++

The 3D MR quality assessment tool uses a pretrained Tensorflow model.
`Tensorflow <https://www.tensorflow.org/install/pip>`_ needs to be installed to enable this tool.
Check out the orginal `repo <https://github.com/FNNDSC/pl-fetal-brain-assessment>`__ for more details.

..
    TWAI segmentation
    +++++++++++++++++

    The toolkit provides a wrapper of the TWAI segmentation algorithm for T2w fetal brain MRI. 
    You may find more detials of this method in the authors' `repo <https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation>`__. 
    To use this tool, you need to clone their repo and update the path in ``config.py`` (see the comment in ``config.py`` for details). 

Install NeSVoR
^^^^^^^^^^^^^^

The last step is to install NeSVoR itself.

.. code-block:: bash

    pip install -e .
