import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from model import VGGAutoencoder, Vanilla_CAE, SiameseNetwork, SiameseNetwork2 
from helper import imshow, show_plot
from losses import ContrastiveLoss, TripletLoss
from dataloader2 import SiameseNetworkDataset

if __name__ == '__main__':

	# Command Line Arguments
	parser = argparse.ArgumentParser(description = 'Siamese Network Training')
	parser.add_argument('--batch_size',type=int,default=32, help='Input batch size for training')
	parser.add_argument('--epochs',type=int,default=10, help='Number of epochs to train')
	parser.add_argument('--lr_recon','--learning_rate',type=float,default=1e-3, help='Learning rate of reconstruction')
	parser.add_argument('--lr_sim',type=float,default=5e-4, help='Learning rate of simiarity')
	parser.add_argument('--training_dir',type=str,default='train_1/train1', help='Path of train dataset')
	parser.add_argument('--training_csv',type=str,default='../Siamese_training_data - positive_examples(1).csv')
	parser.add_argument('--num_workers',type=int,default=2)
	parser.add_argument('--cuda',action='store_true',default=True)
	parser.add_argument('--shuffle', action='store_true',default=False)


	args = parser.parse_args()


	# Intializing device
	device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


	train_transforms = transforms.Compose([
		transforms.ToTensor(),
	])



	# Load the the dataset from raw image folders
	siamese_dataset = SiameseNetworkDataset(args.training_csv,args.training_dir,transform=train_transforms)


	# Load the dataset as pytorch tensors using dataloader
	train_dataloader = DataLoader(siamese_dataset,shuffle=args.shuffle,num_workers=args.num_workers, batch_size=args.batch_size)
	
	copyOfLr = args.lr

#	model = SiameseNetwork2().to(device)
#	model = SiameseNetwork().to(device)
#	model = Vanilla_CAE().to(device)
	model = VGGAutoencoder().to(device)
#	model = VggSiamese().to(device)

#	print(model)

	# Declare loss function
	similarity = ContrastiveLoss()
	criterion = nn.MSELoss()

	# Declare optimizer
	
	optimizer1 = optim.Adagrad(model.parameters(),lr=args.recon)
	optimizer2 = optim.Adam(model.parameters(),lr=args.sim)

	counter = []
	loss_history = []
	iteration_number = 0

	for epoch in range(args.epochs):
		for i, data in tqdm(enumerate(train_dataloader,0)):
			img0, img1 , label = data
			img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)

			optimizer1.zero_grad()
			
			output1, output2, recon1, recon2 = model(img0,img1)

			reconstruction1 = criterion(recon1,img0)
			reconstruction2 = criterion(recon2,img1)
			reconstruction = (reconstruction1 + reconstruction2)/2
			optimizer1.step()


			optimizer1.zero_grad()
			loss_contrastive = similarity(output1,output2,label)
			print("\nContrastive Loss: ", loss_contrastive.item())			
			train_loss = reconstruction + 0.000001 * loss_constrastive
			train_loss.backward()
			optimizer2.step()
			train_loss += train_loss.item()
#			loss_history.append(loss_contrastive)
#			counter.append(epoch)



		# compute the epoch training loss
		loss = train_loss / len(train_dataloader)
		#loss_history.append(loss)
		#counter.append(epoch)


		# display the epoch training loss
		print("\nepoch : {}/{}, contrastive loss = {:.8f}".format(epoch + 1, args.epochs, loss))
		loss_history.append(loss)
		counter.append(epoch)



	show_plot(counter,loss_history)


	torch.save(model.state_dict(),"./Siamese_model_20epochs.pth")
	print("Model saved successfully!")
	#show_plot(counter,loss_history)

	model.eval()
