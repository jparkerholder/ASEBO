#### ASEBO Code

Here is the code to run ASEBO v1. This version of the algorithm runs vanilla ES for *k* steps, appending all of the gradients to a buffer. After k steps, it does PCA on the buffer, ranks the eigenvectors by their corresponding eigenvalues and takes the top ones to form a subspace. The subsequent samples are taken from a covariance matrix comprised of the subspace and the identity at a ratio depending on the bias of the previous subspace. The number of samples is related to the dimensionality of the subspace (related to the eigenvalue decay).

In this setup we have a Toeplitz policy with two hidden layers. The code is setup to run on a local machine, so is not distributed, but it could easily be made to run in parallel eg with ray, since it was adapted from ARS.

The code runs as follows:

```
python train.py --env_name HalfCheetah-v2 --h_dim 32 --k 140

```

The only parameters you need to care about are:
  * k = Generally k upper bounds the number of samples you can subsequently take, so you need it to be larger for bigger. environments (eg 70 for Swimmer/Hopper, 140 for HalfCheetah/Walker2d).
  * h_dim = Typically I use 16 for Swimmer/Hopper, 32 for HalfCheetah/Walker2d, 64 for Ant, 128 for Humanoid (mixed success)
  * sigma/learning_rate = As with all ES algorithms, may need to be tuned.

That is all!! Please let me know if you have questions: jackph@robots.ox.ac.uk
