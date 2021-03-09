# RaspMRI

**RaspMRI** 🍓 is a work in progress biomedical image analysis pipeline to process raw T2 MRI volumes. It involves a deep learning based automatic brain segmentation, a robust co-registration method and an atlas based ischemic stroke severity evaluation pipeline.

- Brain masks can be created with a deep neural network using raw T2 recordings. The U-net architecture created by CAMRIatUNC is further trained on a 7T T2WI dataset containing  MRI volumes of 47 animals with MCAO-induced ischemic stroke. `RodentMRISkullStripping` package has been used for some aspects of the U-net training and the brain mask prediction.


- The segmented brains are co-registered with a template/atlas pair (SIGMA atlas | ref.: https://www.nature.com/articles/s41467-019-13575-7) to enable further anatomical region based segmentation and evaluation. Algorithms implemented in the`SimpleITK` package are used in the co-registration pipeline.


- Regional intensity maps can be calculated and exported for statistical analysis.

...
