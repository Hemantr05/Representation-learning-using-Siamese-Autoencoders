import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F


import torchvision
from torchvision import transforms


from losses import ContrastiveLoss

from dataloader2 import SiameseNetworkDataset
from model import VGGAutoencoder
from helper import imshow, show_plot


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Siamese Network Testing')
	parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing')
	parser.add_argument('--num_workers', type=int, default=6, help='Assign number of worker to the dataloader')
	parser.add_argument('--test_csv',type=str,help='path to test csv')
	parser.add_argument('--test_dir',type=str,help='path to test dataset')

	args = parser.parse_args()

	valid_transforms = transforms.Compose([
		transforms.ToTensor(),
	])


	# Load the saved model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = VGGAutoencoder()
	model.load_state_dict(torch.load("Siamese_model.pth"))


	# Load the test dataset
	test_dataset = SiameseNetworkDataset(training_csv=args.test_csv,training_dir=args.test_dir,
                                                transform=valid_transforms
						)
	test_dataloader = DataLoader(test_dataset,num_workers=args.num_workers,batch_size=args.batch_size,shuffle=True)




	for image in test_dataloader:

		test_image1,test_image2, label = image
		
		concatenated = torch.cat((test_image1, test_image2),0)
		output1,output2  = model(test_image1,test_image2)

		criterion = ContrastiveLoss()
		pairwise_distance = F.pairwise_distance(output1, output2)
		contrastive_loss = criterion(output1, output2,label)

	torchvision.utils.save_image(output1.data, 'output1.png')
	torchvision.utils.save_image(output2.data, 'output2.png')
	torchvision.utils.save_image(pairwise_distance.data, 'pairwise_distance.png')

	torchvision.utils.save_image(test_image2.data, 'orig2.png')
	torchvision.utils.save_image(test_image1.data, 'orig1.png')

