import numpy as np
import matplotlib.pyplot as plt



def imshow(img, text=None,should_save=False):

  npimg = img.cpu().numpy()
#  print(npimg)
  plt.axis("off")
  if text:
    plt.text(75, 8, text, style='italic',fontweight='bold', bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()
    plt.savefig("images/Similarity_output.png")

def show_plot(iteration, loss):
  plt.plot(iteration, loss)
  #plt.show()
  plt.savefig("./TrainLoss.png")


