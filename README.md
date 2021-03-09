# RaspMRI

**RaspMRI** 🍓 is a work in progress biomedical image analysis pipeline to process raw T2 MRI volumes. It involves deep learning based automatic brain segmentation, a robust co-registration method and an atlas based ischemic stroke severity evaluation pipeline.

- Brain masks are created in the raw T2 recordings by deep learning algorithms. The U-net architecture created by CAMRIatUNC is trained on a 7T T2WI dataset consisting of 47 animals. `RodentMRISkullStripping` package has been used for some aspects of the U-net training and predictions.


- The segmented brains are co-registered with a template/atlas pair (SIGMA atlas | ref.: https://www.nature.com/articles/s41467-019-13575-7) to enable further evaluation based on anatomical regions. The co-registration pipeline is  using registration algorithms from the`SimpleITK` package.


- Regional intensity mappings are calculated and can be exported for statistical analysis.

...
