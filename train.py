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
#from models import VggSiamese
from helper import imshow, show_plot
from losses import ContrastiveLoss, TripletLoss
from dataloader2 import SiameseNetworkDataset
#from dataloader import *

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
		#transforms.RandomResizedCrop(size=256,scale=(0.8,1.0)),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		#transforms.CenterCrop(size=224),
		#transforms.Resize((224,224)),
#		transforms.Normalize(mean=[0.485,0.456,0.406],	# imagenet mean/std
#	              std=[0.229,0.224,0.225])
	])



	# Load the the dataset from raw image folders
	siamese_dataset = SiameseNetworkDataset(args.training_csv,args.training_dir,transform=train_transforms)

	# Viewing the sample of images and to check whether its loading properly
    	#vis_dataloader = DataLoader(siamese_dataset, shuffle=args.shuffle, batch_size=args.batch_size)
	#dataiter = iter(vis_dataloader)


	#example_batch = next(dataiter)
	#concatenated = torch.cat((example_batch[0],example_batch[1]),0)
	#imshow(torchvision.utils.make_grid(concatenated))
	#print(example_batch[2].numpy())

	# Load the dataset as pytorch tensors using dataloader
	train_dataloader = DataLoader(siamese_dataset,shuffle=args.shuffle,num_workers=args.num_workers, batch_size=args.batch_size)
   	 #train_loader = get_train_valid_loader(args.training_dir, 32, True)
	
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
	#optimizer = optim.RMSprop(model.parameters(),lr=copyOfLr,alpha=0.99,eps=1e-8,weight_decay=0.0005,momentum=0.9)
	
	optimizer1 = optim.Adagrad(model.parameters(),lr=args.recon)
	optimizer2 = optim.Adam(model.parameters(),lr=args.sim)

#	optimizer = optim.SGD(model.parameters(),lr=copyOfLr,momentum=0.9)
	#ep = [1]

	counter = []
	loss_history = []
	iteration_number = 0

	for epoch in range(args.epochs):
		for i, data in tqdm(enumerate(train_dataloader,0)):
			img0, img1 , label = data
			img0, img1 , label = img0.to(device), img1.to(device) , label.to(device)
			optimizer1.zero_grad()
#			print("\nImage1 size: ", img0.size())
#			print("\nImage2 size: ", img1.size())
#			print("\nLabel size: ", label.size())
#			img0 = img0.view([args.batch_size, 3, 352, 640])
#			img1 = img1.view([args.batch_size, 3, 352, 640])
			output1, output2, recon1, recon2 = model(img0,img1)
#			print("\nOutput1 size: ", output1.size())
#			print("\nOutput2 size: ", output2.size())
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


            #print("\nEpoch: {} | Temp_Loss: {}" .format(epoch,loss))
            # plt.plot(epoch,loss,'r-')
            # plt.xlabel('epochs')
            #plt.ylabel('Training loss')
		# compute the epoch training loss
		loss = train_loss / len(train_dataloader)
		#loss_history.append(loss)
		#counter.append(epoch)


		# display the epoch training loss
		print("\nepoch : {}/{}, contrastive loss = {:.8f}".format(epoch + 1, args.epochs, loss))
		loss_history.append(loss)
		counter.append(epoch)

			#if i %10 == 0 :
			#	print("\nEpoch number {} | Current loss {}\n".format(epoch,loss_contrastive.item()))
			#	iteration_number +=10
			#	counter.append(iteration_number)
			#	loss_history.append(loss_contrastive.item())

	show_plot(counter,loss_history)

#			print("\nImage {}/{} | Train_Loss: {}".format(i,len(train_dataloader),loss_contrastive))


			# add the mini-batch training loss to epoch loss
#			loss_contrastive += loss_contrastive.item()



		# compute the epoch training loss
#		loss = loss / len(train_loader)
#		loss_history.append(loss)
#		counter.append(epoch)


	# display the epoch training loss
#	print("\nepoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, args.epochs, loss))


	torch.save(model.state_dict(),"./Siamese_model_20epochs.pth")
	print("Model saved successfully!")
	#show_plot(counter,loss_history)

	model.eval()
