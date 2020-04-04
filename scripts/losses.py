import torch
import torch.nn as nn
from torch.nn import functional as F


# Contrastive loss


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin


    def forward(self, output1, output2, label):
        #euclidean_distance = F.pairwise_distance(output1, output2,keepdim=True)
        #pdist = nn.PairwiseDistance(p=2)
        #euclidean_distance = pdist(output1,output2)
        euclidean_distance = torch.dist(output1,output2)
#        print("\nEuclidean Distance: ",euclidean_distance)
#        print("\n1-label = ", (1-label))
#        print("\n Distance sqaure: ",torch.pow(euclidean_distance, 2))
#        print("\nmargin - distance: ",(self.margin - euclidean_distance))
#        print("\nclamp of (margin - distance): ", torch.clamp(self.margin - euclidean_distance,min=0.0))
#        print("\nclamp^2: ",torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +  (label) * torch.pow(torch.clamp(self.margin - euclidean_distance,min=0.0), 2))
#        print("\nloss_contrastive: ", loss_contrastive)

        return loss_contrastive


# Triplet loss
    """
    https://arxiv.org/pdf/1503.03832.pdf
    """


class TripletLoss(torch.nn.Module):
    def __init__(self,alpha=2.0):             # alpha is the margin
        super(TripletLoss,self).__init__()
        self.alpha = alpha
        
    def forward(self,output, output1, output2):
        euclidean_dist1 = F.pairwise_distance(output,output1)
        euclidena_dist2 = F.pairwise_distance(output,output2)
        #loss_triplet = torch.max(0,(self.alpha + eucliden_dist1 - euclidean_dist2))
        loss_triplet = torch.max(torch.pow(euclidean_dist1,2) - torch.pow(euclidean_dist2,2) + self.alpha)


# MSE Reconstruction Loss

class MSEReconLoss(nn.Module):
    def __init__(self):
        super(MSEReconLoss, self).__init__()

    def forward(self, input1, input2):
        output1 = nn.MSELoss(input1)
        output2 = nn.MSELoss(input2)

        return output1, output2
