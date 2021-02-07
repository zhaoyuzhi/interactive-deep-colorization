from data import colorize_image as CI
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
import os

def color_scribble(input_path, scribble_path):
	# Choose gpu to run the model on
	gpu_id = 0
	# Initialize colorization class
	colorModel = CI.ColorizeImageCaffe(Xd=256)
	# Load the model
	colorModel.prep_net(gpu_id,'./models/reference_model/deploy_nodist.prototxt','./models/reference_model/model.caffemodel')
	# Load the image
	colorModel.load_image(input_path) # load an image

	# initialize with no user inputs
	#input_ab = np.zeros((2,256,256))
	#mask = np.zeros((1,256,256))
	
	Scribble = cv2.imread(scribble_path, 1)
	#print('scribble_path', scribble_path)
	Scribble = cv2.resize(Scribble, (256, 256))
	Scribble = cv2.cvtColor(Scribble, cv2.COLOR_BGR2RGB)
	img_lab = color.rgb2lab(Scribble).transpose((2, 0, 1))
	mask = np.clip(img_lab[[0], :, :] * 100, 0, 1)
	input_ab = img_lab[1:, :, :]
	
	# call forward
	img_out = colorModel.net_forward(input_ab,mask)

	# get mask, input image, and result in full resolution
	mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
	img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
	img_out_fullres = colorModel.get_img_fullres() # get image at full resolution
	
	return mask_fullres, img_in_fullres, img_out
	


input_paths = '/home/ztt/lty/ILSVRC2012_val_256_SGN_denoised(second_step)'
scribble_paths = '/home/ztt/lty/ILSVRC2012_val_256_colorscribble'
output_paths = '/home/ztt/lty/hint_ILSVRC2012_val_256_SGN_denoised(second_step)'
count = 0

flies = os.listdir(input_paths) 

for filename in flies:
	if count < -1:
		print(count)
		count += 1
	else:
		input_path = input_paths + '/' + filename
		pngfilename = filename.split('.')[0]
		scribble_path = scribble_paths + '/' + pngfilename + '.png'
		mask_fullres_path = output_paths + '/mask_' + filename
		img_in_fullres_path = output_paths + '/in_' + filename
		img_out_fullres_path = output_paths + '/' + filename
		#print(input_path)
		mask_fullres, img_in_fullres, img_out_fullres = color_scribble(input_path, scribble_path)
		img_out_fullres = cv2.cvtColor(img_out_fullres, cv2.COLOR_BGR2RGB)
		Scribble = cv2.imread(scribble_path, 1)

		#cv2.imwrite(mask_fullres_path, Scribble)
		#cv2.imwrite(img_in_fullres_path, img_in_fullres)
		cv2.imwrite(img_out_fullres_path, img_out_fullres)
		print(count)
		count += 1

