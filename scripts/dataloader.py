import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

# Custom dataloader
class SiameseNetworkDataset(Dataset):
    def __init__(self, training_csv=None,training_dir=None, transform=None):
      self.training_df = pd.read_csv(training_csv)
      self.training_df.columns = ["frame1","frame2","label"]
      self.training_df["frame1"] = self.training_df["frame1"].apply(str) + ".jpg"
      self.training_df["frame2"] = self.training_df["frame2"].apply(str) + ".jpg"
      self.training_dir = training_dir
      self.transform = transform

    def __getitem__(self,index):

      # Getting the image path
      image1_path = os.path.join(self.training_dir,(self.training_df).iat[index,0])
      image2_path = os.path.join(self.training_dir,(self.training_df).iat[index,1])


      # Loading the image
      img0 = Image.open(image1_path)
      img1 = Image.open(image2_path)
      #img0 = img0.convert("L")
      #img1 = img1.convert("L")

		
      # Apply image transformations
      if self.transform is not None:
        img0 = self.transform(img0)
        img1 = self.transform(img1)
		
      label = torch.from_numpy(np.array([int(self.training_df.iat[index,2])],dtype=np.float32))
#      print("Label: ",label)

      return img0, img1, label

    def __len__(self):
      return len(self.training_df)
