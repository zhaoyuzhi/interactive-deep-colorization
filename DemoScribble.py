from data import colorize_image as CI
import matplotlib.pyplot as plt
import numpy as np

# Choose gpu to run the model on
gpu_id = 0

# Initialize colorization class
colorModel = CI.ColorizeImageCaffe(Xd=256)

# Load the model
colorModel.prep_net(gpu_id,'./models/reference_model/deploy_nodist.prototxt','./models/reference_model/model.caffemodel')

# Load the image
colorModel.load_image('./test_imgs/bird_gray.jpg') # load an image

def put_point(input_ab,mask,loc,p,val):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       half-patch size
    # val         2 tuple      (a,b) value of user input
    input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
    mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
    print('*********val********', np.array(val)[:,np.newaxis,np.newaxis])
    print('*********val shape********', (np.array(val)[:,np.newaxis,np.newaxis]).shape)
    print('*********img_l max********', np.max(mask))
    print('*********img_l min********', np.min(mask))
    print('*********img_ab max********', np.max(input_ab))
    print('*********img_ab min********', np.min(input_ab))
    return (input_ab,mask)
# initialize with no user inputs
input_ab = np.zeros((2,256,256))
mask = np.zeros((1,256,256))

# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[160,130],3,[-30,60])
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[100,100],3,[-20,70])
# add a blue point in the middle of the image
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[200,130],3,[-30,60])
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[10,100],3,[-20,70])
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[250,130],3,[60,60])
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[160,230],3,[-30,60])
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[100,200],3,[-20,70])
# add a blue point in the middle of the image
(input_ab,mask) = put_point(input_ab,mask,[150,230],3,[60,60])
(input_ab,mask) = put_point(input_ab,mask,[150,130],3,[60,60])

# call forward
img_out = colorModel.net_forward(input_ab,mask)

# get mask, input image, and result in full resolution
mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
img_out_fullres = colorModel.get_img_fullres() # get image at full resolution

# show user input, along with output
plt.figure(figsize=(10,6))
plt.imshow(np.concatenate((mask_fullres,img_in_fullres,img_out_fullres),axis=1));
plt.title('Mask of user points / Input grayscale with user points / Output olorization')
plt.axis('off');
plt.savefig('./Scribble results/with user points.png')





