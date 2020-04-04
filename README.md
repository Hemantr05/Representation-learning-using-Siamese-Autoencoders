# Representation learning using Siamese Autoencoders

The aim focus of is to improve the pretext task of object detection with the help of coherence-metrics.
This implementation acheives this by training the Siamese Autoencoder (`models/model.py`) on a dataset with the help of the `scripts/dataloader.py`. The model works on the principle of attention mechanism, where it learns spatial features with each reconstruction.

The network is uses two loss functions (in `scripts/losses.py`): ContrastiveLoss and MSELoss.
1. The ConstrastiveLoss is used to measure the similarity between two frames from the dataloader (to measure coherence metrics)

2. The MSELoss is used for reconstruction of the inputs (for the encoder to learn representations)



# Usage:

`

# Coming soon

`


References:
1. [Unsupervised Learning of Spatiotemporally Coherent Metrics](https://arxiv.org/abs/1412.6056)


