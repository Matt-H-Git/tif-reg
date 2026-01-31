# Preprocessing

Datasets should handle any preprocessing of the data, 
any training loop or inference assumes the data has already been preprocessed.

The best preprocessing for tif-reg is to mean centre the data, then scale the cloud to a unit sphere.
This should be done before generating a transform and obtaining the source cloud.

