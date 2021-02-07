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
	
	return mask_fullres, img_in_fullres, img_out_fullres
	
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def define_test_list(imglist, classlist):
	# Calculate the whole number of each class
	imgroot = [list() for i in range(len(classlist))]
	for i, classname in enumerate(classlist):
		for j, imgname in enumerate(imglist):
			if imgname.split('/')[-2] == classname:
				imgroot[i].append(imgname)
	return imgroot

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# run on tiantian's computer, for SVCNet paper
if __name__ == '__main__':
	
	input_paths = '/home/ztt/lty/interactive-deep-colorization-master/DAVIS-Videvo-test/DAVIS-gray'
	scribble_paths = '/home/ztt/lty/interactive-deep-colorization-master/DAVIS-Videvo-test/fixed_color_scribbles/DAVIS'
	output_paths = '/home/ztt/lty/interactive-deep-colorization-master/DAVIS-Videvo-result/DAVIS'
	imglist = text_readlines('./DAVIS-Videvo-test/DAVIS_test_imagelist.txt')
	classlist = text_readlines('./DAVIS-Videvo-test/DAVIS_test_class.txt')

	input_paths = '/home/ztt/lty/interactive-deep-colorization-master/DAVIS-Videvo-test/videvo-gray'
	scribble_paths = '/home/ztt/lty/interactive-deep-colorization-master/DAVIS-Videvo-test/fixed_color_scribbles/videvo'
	output_paths = '/home/ztt/lty/interactive-deep-colorization-master/DAVIS-Videvo-result/videvo'
	imglist = text_readlines('./DAVIS-Videvo-test/videvo_test_imagelist.txt')
	classlist = text_readlines('./DAVIS-Videvo-test/videvo_test_class.txt')

	imgroot = define_test_list(imglist, classlist)

	for i in range(len(imgroot)):
		for j in range(len(imgroot[i])):
			classname = imgroot[i][j].split('/')[-2]
			imgname = imgroot[i][j].split('/')[-1]
			check_path(os.path.join(output_paths, classname))

			input_path = os.path.join(input_paths, classname, imgname)
			scribble_path = os.path.join(scribble_paths, classname, imgname.split('.')[0] + '.png')

			mask_fullres_path = os.path.join(output_paths, classname, 'mask_' + imgname)
			img_in_fullres_path = os.path.join(output_paths, classname, 'in_' + imgname)
			img_out_fullres_path = os.path.join(output_paths, classname, imgname)

			'''
			print(input_path)
			print(scribble_path)
			print(len(imglist))
			'''

			mask_fullres, img_in_fullres, img_out_fullres = color_scribble(input_path, scribble_path)
			img_out_fullres = cv2.cvtColor(img_out_fullres, cv2.COLOR_BGR2RGB)
			Scribble = cv2.imread(scribble_path, 1)

			cv2.imwrite(mask_fullres_path, Scribble)
			cv2.imwrite(img_in_fullres_path, img_in_fullres)
			cv2.imwrite(img_out_fullres_path, img_out_fullres)
