import torch
from utils import *
from train import *


colours_dic_addr = 'car_colours_kmeans24.npy'
your_student_id = 400443183

# other constants if needed
##############################################################################################
#                                 define some arguments if needed to pass                    #
##############################################################################################
# Download CIFAR dataset
(x_train, y_train), (x_test, y_test) = load_cifar10()

# LOAD THE COLOURS CATEGORIES
colours = np.load(colours_dic_addr, allow_pickle=True, encoding='bytes')


print ("\n\nLength Car Colors: ",len(colours),"\t\t" , "Shape Car Colors: " ,colours.shape ,"\n\n")
print ("x_train: " , x_train.shape , "\t  y_train: " , y_train.shape , "\n")
print ("x_test: " , x_test.shape , "\t  y_test: " , y_test.shape , "\n")


plt.imshow(torch.from_numpy(x_train[212]).permute(1,2,0))

import torch
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
args = AttrDict()
args_dict = {
              'gpu':True,
              'num_filters':64,
              'seed':your_student_id,
              'category_id': 1,
              'valid':False,
              'checkpoint':"",
              'model':"UNET",
              'kernel':3,
              'learn_rate':0.3,
              'batch_size':100,
              'epochs':50,
              'plot':True,
              'experiment_name': 'colourization_cnn',
              'visualize': False,
              'downsize_input':False,
}
args.update(args_dict)

##############################################################################################
#                                 Call train function                                        #
##############################################################################################
train(args , x_train, y_train, x_test, y_test, colours=colours , model_mode="Base" ,model="Base")