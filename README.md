# Representation learning using Siamese Autoencoders

The aim focus of is to improve the pretext task of object detection with the help of coherence-metrics.
This implementation acheives this by training the Siamese Autoencoder (`models/model.py`) on a dataset with the help of the `scripts/dataloader.py`. The model works on the principle of attention mechanism, where it learns spatial features with each reconstruction.

The network is uses two loss functions (in `scripts/losses.py`): ContrastiveLoss and MSELoss.
1. The ConstrastiveLoss is used to measure the similarity between two frames from the dataloader (to measure coherence metrics)

2. The MSELoss is used for reconstruction of the inputs (for the encoder to learn representations)



# Usage:
(Under progress)

## Training:

Enable gpu 0

`$export CUDA_VISIBLE_DEVICES=0`

Train network

`$CUDA_VISIBLE_DEVICES=0 python train.py --lr_sim 5e-4 --lr_recon 1e-4 --epochs 25 --batch_size 4 --mu 1e-5 --training_dir 'Add dataset path' --training_csv 'Add corresponding csv path --num_workers 2'`

## Testing:
`$CUDA_VISIBLE_DEVICES=0 python test.py --test_dir 'Add-path-to-testset' --test_csv 'Add-path-to-csv.csv'`




References:
1. [Unsupervised Learning of Spatiotemporally Coherent Metrics](https://arxiv.org/abs/1412.6056)


