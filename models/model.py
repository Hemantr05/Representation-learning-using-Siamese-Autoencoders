import torch
import torch.nn as nn

import torchvision
from torchvision import models

# Loading pretrained model
vgg16 = models.vgg16(pretrained=True)
vgg16_features = vgg16.features

# Freeze model weights
for param in vgg16_features.parameters():
    param.requires_grad = True

# Using the features layers only
model = vgg16_features[0:29]



# VGG16 Autoencoder
class VGGAutoencoder(nn.Module):
    def __init__(self):
        super(VGGAutoencoder, self).__init__()
        
        self.encoder = model
        
        #self.bottleneck = nn.Sequential(
        #                    nn.Linear(in_features=512,out_features=256),
        #                    nn.Linear(in_features=256,out_features=512)
        #                )
#        self.embedding = nn.Sequential( nn.Linear(512*22*40,256),
#                                        nn.ReLU(inplace=True),
#                                        nn.Linear(256,512),
#                                        nn.ReLU(inplace=True)
#                                      )

        
        self.decoder = nn.Sequential(
                            #nn.UpsamplingBilinear(scale_factor=2),
                            nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.UpsamplingBilinear2d(scale_factor=2),
                            nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(512,512,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.UpsamplingBilinear2d(scale_factor=2),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(512,256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
#                           nn.UpsamplingNearest2d(scale_factor=1),
                            nn.UpsamplingBilinear2d(scale_factor=2),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(256,256,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(256,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(128,128,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.UpsamplingBilinear2d(size=(352,640)),
                            nn.ConvTranspose2d(128,64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(64,64,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose2d(64,3,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                            #nn.ReLU(inplace=True),
                            #nn.ConvTranspose2d(3,1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
                        )

#    def latent_space(self,features):
#        self.fc1 = nn.Linear(features.size(),256)
#        self.fc2 = nn.Linear(256,features.size())
        
    def forward_once(self,x):
#        print("Image input to model: ",x.size())
        enc_out = self.encoder(x)
        print("\nEncoder output size: ",enc_out.size())
        print(enc_out)
        #embedding = enc_out.view(enc_out.size()[])
#        embedding_out = self.embedding(enc_out)
#        embedding = torch.flatten(enc_out)
#        embedding = torch.flatten(enc_out)
#        print("\nFlatten encoder size: ",embedding.size())
#        enc_out = self.latent_space(enc_out)
#        embedding = enc_out.view(enc_out.size(),256)       
#        embedding = embedding.view(256,512)
#        embedding = embedding.view(512,-1)
#        embedding = embedding.flatten()
#        embedding = enc_out.view(enc_out.size()[1],512)

#        print("\nOutput of the embedding1: ",embedding.size())

#        embedding1 = nn.Linear(512*22*40,256)(enc_out)
#        embedding = nn.Linear(256,512)(embedding1)
        #embedding = nn.Linear(,256)(embedding)

       # out = self.bottleneck(out)
        #embedding_out = embedding.view([args.batch_size,3,22,45])
#        embedding = embedding.view(-1,512)
#        print("\nOuput of the embedding2: ", embedding.size())

#        embedding = enc_out.view(512,512)
#        print("\nOutput of the last embedding: ",embedding.size())
#        embedding = torch.flatten(enc_out)
#        print("\nEmbedding size: ", embedding.size())
#        embedding.view(-1,512)
#        print("\nEmbedding size: ", embedding.size())
        dec_out = self.decoder(enc_out)
        print("\ndecoder output: ",dec_out.size())

        return torch.flatten(enc_out), dec_out
    

    def forward(self,input1,input2):
#        print("\ninput1 size: ",input1.size())

        # Forward pass on input1
        output1, recon1 = self.forward_once(input1)
#        print("\nOutput1 size: ", output1.size())
        #output1.view(output1.size(),256)
#        print("\nParameters1 ", output1.parameters())

        # Forward pass on input2
        output2, recon2 = self.forward_once(input2)
#        print("\noutput2 size:",output2.size())
        #output2.view(output2.size(),256)
#        print("\nEmbedding of input2: ", output2.size())
       
        return output1, output2




class Vanilla_CAE(nn.Module):
        def __init__(self):
                super(Vanilla_CAE,self).__init__()

                self.encoder = nn.Sequential(
                                        nn.Conv2d(3,16,3,stride=1,padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,stride=2)
                                        )

                self.decoder = nn.Sequential(
                                        nn.ConvTranspose2d(16,16,3,stride=2,padding=1,output_padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16,3,3,stride=1,padding=1,output_padding=0),
                                        nn.ReLU()
                                        )

        def forward_once(self,x):
                enc_out = self.encoder(x)
#                print("\nencoder output size:", enc_out.size())
                dec_out = self.decoder(enc_out)
#                print("\ndecoder output size:", dec_out.size())

                return dec_out

        def forward(self,input1,input2):
                output1 = self.forward_once(input1)
#                print("\nOutput1 size: ",output1.size())

                output2 = self.forward_once(input2)
#                print("\nOutput2 size: ", output2.size())

                return output1, output2


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output1 = self.cnn1(x)
        #output = output.view(output.size()[0], -1)
        #output = output.view(output.size()[0]) 
        output2 = self.fc1(output1)
        return output1 ,output2

    def forward(self, input1, input2):
        out1, out2 = self.forward_once(input1)
        out1, out2 = self.forward_once(input2)
        return out1, out2


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            Flatten(),
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
