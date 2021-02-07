import caffe
from caffe_files.caffe_traininglayers import *
from caffe_files.color_quantization import *
from caffe_files.util import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from data import colorize_image as CI
from skimage import color
from data import lab_gamut as lab

# Colorization with reference global histogram
def get_global_histogram(ref_path):
	ref_img_fullres = caffe.io.load_image(ref_path)
	img_glob_dist = (255*caffe.io.resize_image(ref_img_fullres,(Xd,Xd))).astype('uint8') # load image
	gt_glob_net.blobs['img_bgr'].data[...] = img_glob_dist[:,:,::-1].transpose((2,0,1)) # put into 
	gt_glob_net.forward();
	glob_dist_in = gt_glob_net.blobs['gt_glob_ab_313_drop'].data[0,:-1,0,0].copy()
	return (glob_dist_in,ref_img_fullres)
	
def color_global(input_path, ref_path):

	gpu_id = 0 # gpu to use
	Xd = 256

	# Colorization network
	cid = CI.ColorizeImageCaffeGlobDist(Xd)
	cid.prep_net(gpu_id,prototxt_path='./models/global_model/deploy_nodist.prototxt',\
				caffemodel_path='./models/global_model/global_model.caffemodel')

	# Global distribution network - extracts global color statistics from an image
	gt_glob_net = caffe.Net('./models/global_model/global_stats.prototxt',\
						   './models/global_model/dummy.caffemodel', caffe.TEST)


	# Load image
	cid.load_image(input_path)

	# Dummy variables
	input_ab = np.zeros((2,Xd,Xd))
	input_mask = np.zeros((1,Xd,Xd))

	# Colorization without global histogram
	img_pred = cid.net_forward(input_ab,input_mask);
	img_pred_auto_fullres = cid.get_img_fullres()

	# Gray image
	img_gray_fullres = cid.get_img_gray_fullres()


	(glob_dist_ref,ref_img_fullres) = get_global_histogram(ref_path)
	img_pred = cid.net_forward(input_ab,input_mask,glob_dist_ref);
	img_pred_withref_fullres = cid.get_img_fullres()

	return img_pred_withref_fullres


# image to colorize
input_path = './test_imgs/bird_gray.jpg'
out_path = './test_imgs/out.png'
# color histogram to use
ref_path = './test_imgs/global_ref_bird/ILSVRC2012_val_00048203.JPEG'
img_pred_withref_fullres = color_global(input_path, ref_path)

img_pred_withref_fullres = cv2.cvtColor(img_pred_withref_fullres, cv2.COLOR_BGR2RGB)
cv2.imwrite(out_path, img_pred_withref_fullres)



