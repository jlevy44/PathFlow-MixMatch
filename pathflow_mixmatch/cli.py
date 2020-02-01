import fire
import numpy as np, cv2
import cv2
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import label as scilabel, distance_transform_edt
import scipy.ndimage as ndimage
from skimage import morphology as morph
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
import pandas as pd
from skimage import data, util, measure
from skimage.transform import rotate, resize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn
import SimpleITK as sitk
import sys
import os
import time
import matplotlib.pyplot as plt
import torch as th
import warnings
warnings.filterwarnings("ignore")
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al
import matplotlib
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
	with open(os.devnull, "w") as devnull:
		old_stdout = sys.stdout
		sys.stdout = devnull
		try:
			yield
		finally:
			sys.stdout = old_stdout


def label_objects(I, min_object_size, threshold=220, connectivity=8, kernel=8, apply_watershed=False):

	#try:
	BW = (cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)<threshold).astype(bool)
	#     if apply_watershed:
	#         BW = morph.binary_opening(BW, np.ones((connectivity,connectivity)).astype(int))
	labels = scilabel(BW)[0]
	BW = fill_holes(morph.remove_small_objects(labels, min_size=min_object_size, connectivity = connectivity, in_place=True))
	if apply_watershed:
		distance = distance_transform_edt(BW)
		local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((kernel, kernel)), labels=BW)
		markers = scilabel(local_maxi)[0]
		labels = watershed(-distance, markers, mask=BW)
	else:
		labels = scilabel(BW)[0]
	return(BW!=0),labels

def get_matched_tissue(props,props2):
	x=PCA(n_components=4,random_state=42).fit_transform(pd.concat((props,props2)))#n_components=2 StandardScaler().fit_transform()
	return enumerate(pd.DataFrame(sklearn.metrics.pairwise.pairwise_distances(x[:props.shape[0]],x[props.shape[0]:],'euclidean')).values.argmin(1))#pd.DataFrame(sklearn.metrics.pairwise.cosine_similarity(pd.DataFrame(x).T)).iloc[:props.shape[0],props.shape[0]:]#pd.DataFrame(x).T.corr()>0.95

def displace_image(img, displacement, gpu_device):
	channels=[]
	for i in range(3):
		im=sitk.GetImageFromArray(img[...,i])
		im=al.utils.image.create_tensor_image_from_itk_image(im, dtype=th.float32, device='cuda:{}'.format(gpu_device))
		channels.append(al.transformation.utils.warp_image(im, displacement).numpy())
	return np.uint8(np.stack(channels).transpose((1,2,0)))

# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def affine_register(im1, im2, iterations=1000, lr=0.01, transform_type='similarity', gpu_device=0):
	assert transform_type in ['similarity', 'affine', 'rigid']
	start = time.time()

	# set the used data type
	dtype = th.float32
	# set the device for the computaion to CPU
	device = th.device("cuda:{}".format(gpu_device))

	# In order to use a GPU uncomment the following line. The number is the device index of the used GPU
	# Here, the GPU with the index 0 is used.
	# device = th.device("cuda:0")

	# load the image data and normalize to [0, 1]
    # add mask to loss function
	fixed_image = al.utils.image.create_tensor_image_from_itk_image(sitk.GetImageFromArray(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)), dtype=th.float32, device=device)#al.read_image_as_tensor("./practice_reg/1.png", dtype=dtype, device=device)#th.tensor(img1,device='cuda',dtype=dtype)#
	moving_image = al.utils.image.create_tensor_image_from_itk_image(sitk.GetImageFromArray(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)), dtype=th.float32, device=device)#al.read_image_as_tensor("./practice_reg/2.png", dtype=dtype, device=device)#th.tensor(img2,device='cuda',dtype=dtype)#

	fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

	# convert intensities so that the object intensities are 1 and the background 0. This is important in order to
	# calculate the center of mass of the object
	fixed_image.image = 1 - fixed_image.image
	moving_image.image = 1 - moving_image.image

	# create pairwise registration object
	registration = al.PairwiseRegistration()

	transforms=dict(similarity=al.transformation.pairwise.SimilarityTransformation,
				   affine=al.transformation.pairwise.AffineTransformation,
				   rigid=al.transformation.pairwise.RigidTransformation)

	# choose the affine transformation model
	transformation = transforms[transform_type](moving_image, opt_cm=True)
	# initialize the translation with the center of mass of the fixed image
	transformation.init_translation(fixed_image)

	registration.set_transformation(transformation)

	# choose the Mean Squared Error as image loss
	image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

	registration.set_image_loss([image_loss])

	# choose the Adam optimizer to minimize the objective
	optimizer = th.optim.Adam(transformation.parameters(), lr=lr, amsgrad=True)

	registration.set_optimizer(optimizer)
	registration.set_number_of_iterations(iterations)

	# start the registration
	registration.start()

	# set the intensities back to the original for the visualisation
	fixed_image.image = 1 - fixed_image.image
	moving_image.image = 1 - moving_image.image

	# warp the moving image with the final transformation result
	displacement = transformation.get_displacement()
	warped_image = al.transformation.utils.warp_image(moving_image, displacement)

	end = time.time()

	print("=================================================================")

	print("Registration done in:", end - start, "s")
	print("Result parameters:")
	transformation.print()

	# plot the results
	plt.subplot(131)
	plt.imshow(fixed_image.numpy(), cmap='gray')
	plt.title('Fixed Image')

	plt.subplot(132)
	plt.imshow(moving_image.numpy(), cmap='gray')
	plt.title('Moving Image')

	plt.subplot(133)
	plt.imshow(warped_image.numpy(), cmap='gray')
	plt.title('Warped Moving Image')
	return displacement, warped_image, transformation._phi_z, registration.loss.data.item()

def get_loss(im1,im2,gpu_device):

	dh=int(np.abs((im1.shape[0]-im2.shape[0])))
	if im1.shape[0]>im2.shape[0]:
		img2 = cv2.copyMakeBorder(im2, dh, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
	elif im1.shape[0]<im2.shape[0]:
		im1 = cv2.copyMakeBorder(im1, dh, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
	dw=int(np.abs((im1.shape[1]-im2.shape[1])))
	if im1.shape[1]>im2.shape[1]:
		im2 = cv2.copyMakeBorder(im2, 0, 0, 0, dw, cv2.BORDER_CONSTANT, value=[255,255,255])
	elif im1.shape[1]<im2.shape[1]:
		im1 = cv2.copyMakeBorder(im1, 0, 0, 0, dw, cv2.BORDER_CONSTANT, value=[255,255,255])

	_, _, _,loss=affine_register(np.uint8(im1), np.uint8(im2), 100, lr=0.01, transform_type='rigid', gpu_device=gpu_device)
	th.cuda.empty_cache()
	return loss

def rotate_detector(im1,im2,gpu_device):
	angles={}
	for k,angle in enumerate([0, 90, 180, 270]):
		im_test=np.rot90(im2,k)#rotate(im2,angle, resize=True, center=None, order=1, mode='constant', cval=255., clip=False, preserve_range=False)
		angles[angle]=get_loss(im1,im_test,gpu_device)
	return angles

def correct_rotation(im1, im2, scaling_factor=4, gpu_device=0):
	print("Resizing image 1 for rotation check.")
	# im1_small=resize(im1, (im1.shape[0] // scaling_factor, im1.shape[1] // scaling_factor),
	# 				   anti_aliasing=True)*255.
	im1_small=cv2.resize(im1,(int(im1.shape[1] // scaling_factor), int(im1.shape[0] // scaling_factor)))
	print("Resizing image 2 for rotation check.")
	# im2_small=resize(im2, (im2.shape[0] // scaling_factor, im2.shape[1] // scaling_factor),
	# 				   anti_aliasing=True)*255.
	im2_small=cv2.resize(im2,(int(im2.shape[1] // scaling_factor), int(im2.shape[0] // scaling_factor)))

	print("Detecting ideal rotation.")
	with suppress_stdout():
		rotation_loss_dict=rotate_detector(im1_small,im2_small,gpu_device)

	th.cuda.empty_cache()

	rotations=np.array(list(rotation_loss_dict.items()))
	angle=int(rotations[np.argmin(rotations[:,1]),0])
	print("Ideal rotation angle: {}.".format(angle))
	return np.rot90(im2,angle//90)

def rotate_image(mat, angle):
	"""
	https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides/47248339
	Rotates an image (angle in degrees) and expands image to avoid cropping
	"""

	height, width = mat.shape[:2] # image shape has 3 dimensions
	image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = np.abs(rotation_mat[0,0])
	abs_sin = np.abs(rotation_mat[0,1])

	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)

	# subtract old image center (bringing image back to origo) and adding the new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]

	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), borderValue=(255,255,255))
	return rotated_mat

# add moments, first image should have 1+ corresponding segments / 2 sections
# while first has fewer sections
# add mask !!!
def register_images_(npy1='A.npy', npy2='B.npy', connectivity=8, apply_watershed=False, overwrite=False, min_object_size=50000, clip=True, fix_rotation=True, mult_factor=2.0, scaling_factor=4., output_dir='practice_reg_results/images', properties=['area','convex_area','eccentricity','major_axis_length','minor_axis_length','inertia_tensor','inertia_tensor_eigvals','perimeter','orientation'], gpu_device=0, max_rotation_vertical_px=0):

	print("Loading images.")

	im1=np.load(npy1)
	im2=np.load(npy2)

	if max_rotation_vertical_px:
		scaling_factor=((sum(im1.shape[:2])+sum(im2.shape[:2]))/4.)/float(max_rotation_vertical_px)
		print("New scaling factor: {}".format(scaling_factor))

	if fix_rotation:
		print("Adjusting 90 degree rotations.")

		im2=correct_rotation(im1, im2, scaling_factor=scaling_factor, gpu_device=gpu_device)

	th.cuda.empty_cache()

	print("Locating Sections.")

	mask,labels=label_objects(im1, connectivity=connectivity, min_object_size=min_object_size, apply_watershed=apply_watershed)
	mask2,labels2=label_objects(im2, connectivity=connectivity, min_object_size=min_object_size, apply_watershed=apply_watershed)

	print("Estimating section properties.")

	props = pd.DataFrame(measure.regionprops_table(labels, cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),properties=properties))
	props2 = pd.DataFrame(measure.regionprops_table(labels2, cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),properties=properties))

	bboxes=pd.DataFrame(measure.regionprops_table(labels,cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY),properties=['bbox']))
	bboxes2=pd.DataFrame(measure.regionprops_table(labels2,cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY),properties=['bbox']))

	print("Matching tissue sections.")

	sections=list(get_matched_tissue(props,props2))

	N=len(sections)

	del im1, im2

	im1=np.load(npy1,mmap_mode='r+')
	im2=np.load(npy2,mmap_mode='r+')

	print("Performing alignments on pairs of sections.")

	for idx,(i,j) in enumerate(sections):

		npy_out1=os.path.join(output_dir,os.path.basename(npy1)).replace('.npy','_{}.png'.format(idx))
		npy_out2=os.path.join(output_dir,os.path.basename(npy2)).replace('.npy','_{}.png'.format(idx))

		if os.path.exists(npy_out1)==False or os.path.exists(npy_out1)==False or overwrite:

			try:

				print("[{}/{}] - Extracting and rotating sections {} and {}".format(idx+1,N,i,j))
				angle=-props.loc[i,'orientation']*180./(np.pi)
				img=im1[bboxes.iloc[i,0]:bboxes.iloc[i,2],bboxes.iloc[i,1]:bboxes.iloc[i,3]].copy()#.copy()
				img[labels[bboxes.iloc[i,0]:bboxes.iloc[i,2],bboxes.iloc[i,1]:bboxes.iloc[i,3]]!=(i+1)]=255.
				img1=rotate_image(img, angle)#np.uint8(rotate(img, -props.loc[i,'orientation']*180./(np.pi), resize=True, center=None, order=1, mode='constant', cval=1., clip=clip, preserve_range=False)*255.)

				angle=-props2.loc[j,'orientation']*180./(np.pi)
				img=im2[bboxes2.iloc[j,0]:bboxes2.iloc[j,2],bboxes2.iloc[j,1]:bboxes2.iloc[j,3]].copy()#.copy()
				img[labels2[bboxes2.iloc[j,0]:bboxes2.iloc[j,2],bboxes2.iloc[j,1]:bboxes2.iloc[j,3]]!=(j+1)]=255.
				img2=rotate_image(img, angle)#np.uint8(rotate(img, -props2.loc[j,'orientation']*180./(np.pi), resize=True, center=None, order=1, mode='constant', cval=1., clip=clip, preserve_range=False)*255.)

				print("[{}/{}] - Trimming sections to proper W & H".format(idx+1,N))

				c=int(img1.shape[1]/2.)
				w=max(int(props.loc[i,'minor_axis_length']/2),int(props2.loc[j,'minor_axis_length']/2*mult_factor))
				h=max(int(img1.shape[0]/2.),int(img2.shape[1]/2.))
				img1=img1[:,c-w:c+w]

				c=int(img2.shape[1]/2.)
				img2=img2[:,c-w:c+w]

				dh=int(np.abs((img1.shape[0]-img2.shape[0])))
				if img1.shape[0]>img2.shape[0]:
					img2 = cv2.copyMakeBorder(img2, dh//2+dh%2, dh//2, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
				elif img1.shape[0]<img2.shape[0]:
					img1 = cv2.copyMakeBorder(img1, dh//2+dh%2, dh//2, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
				dw=int(np.abs((img1.shape[1]-img2.shape[1])))
				if img1.shape[1]>img2.shape[1]:
					img2 = cv2.copyMakeBorder(img2, 0, 0, dw//2+dw%2, dw//2, cv2.BORDER_CONSTANT, value=[255,255,255])
				elif img1.shape[1]<img2.shape[1]:
					img1 = cv2.copyMakeBorder(img1, 0, 0, dw//2+dw%2, dw//2, cv2.BORDER_CONSTANT, value=[255,255,255])

				print("[{}/{}] - Begin alignment of sections.".format(idx+1,N))

				with suppress_stdout():
					new_img=displace_image(img2,affine_register(img1, img2, gpu_device=gpu_device)[0],gpu_device=gpu_device) # new tri, output he as well

				th.cuda.empty_cache()

				print("[{}/{}] - Writing registered sections to file.".format(idx+1,N))

				cv2.imwrite(npy_out1,cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
				cv2.imwrite(npy_out2,cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))

			except Exception as e:
				print(str(e))

class Commands(object):
	def __init__(self):
		pass

	def register_images(self,
							npy1='A.npy',
							npy2='B.npy',
							connectivity=8,
							overwrite=False,
							clip=True,
							min_object_size=50000,
							fix_rotation=True,
							mult_factor=1.8,
							scaling_factor=4.,
							max_rotation_vertical_px=0,
							output_dir='reg_results/images',
							properties=['area',
										'convex_area',
										'eccentricity',
										'major_axis_length',
										'minor_axis_length',
										'inertia_tensor',
										'inertia_tensor_eigvals',
										'perimeter',
										'orientation'],
							gpu_device=0):
		register_images_(npy1=npy1,
							npy2=npy2,
							connectivity=connectivity,
							apply_watershed=False,
							overwrite=overwrite,
							min_object_size=min_object_size,
							clip=clip,
							fix_rotation=fix_rotation,
							mult_factor=mult_factor,
							scaling_factor=scaling_factor,
							output_dir=output_dir,
							properties=properties,
							gpu_device=gpu_device,
							max_rotation_vertical_px=max_rotation_vertical_px)

def main():
	fire.Fire(Commands)

if __name__=='__main__':
	main()
