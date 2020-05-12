import fire
import numpy as np, pandas as pd, cv2
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
import contextlib
warnings.filterwarnings("ignore")
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import airlab as al
import matplotlib
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys, os
import pickle
# from apex import amp


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

def displace_image(img, displacement, gpu_device, dtype=th.float32):
	# channels=[]
	image=al.image_from_numpy(img,(),(), device=('cuda:{}'.format(gpu_device) if gpu_device>=0 else 'cpu'))#[...,i]
	image_size = image.size
	grid = al.transformation.utils.compute_grid(image_size[:2], dtype=image.dtype, device=image.device)
	out=al.image_from_numpy(np.empty(image_size),(),(),device=image.device,dtype=image.dtype)
	if len(image_size)==2:
		out.image =  al.transformation.utils.F.grid_sample(image.image, displacement + grid)
	else:
		for i in range(image_size[-1]):
			out.image[...,i] = al.transformation.utils.F.grid_sample(image.image[...,i], displacement + grid)
	# for i in range(3):
	#
	# 	im=al.utils.image.create_tensor_image_from_itk_image(im, dtype=dtype, device=('cuda:{}'.format(gpu_device) if gpu_device>=0 else 'cpu'))
	# 	channels.append(al.transformation.utils.warp_image(im, displacement).numpy())
	return np.uint8(out.image.numpy())#np.stack(channels).transpose((1,2,0))

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


def affine_register(im1, im2, iterations=1000, lr=0.01, transform_type='similarity', gpu_device=0, opt_cm=True, sigma=[[11,11],[11,11],[3,3]], order=2, pyramid=[[4,4],[2,2]], loss_fn='mse', use_mask=False, interpolation='bicubic', half=False, regularisation_weight=[1,5,50]):
	assert use_mask==False, "Masking not implemented"
	assert transform_type in ['similarity', 'affine', 'rigid', 'non_parametric','bspline','wendland']
	if half:
		raise NotImplementedError
	start = time.perf_counter()

	# set the used data type
	dtype = th.float32# if not half else th.half
	# set the device for the computaion to CPU
	device = th.device("cuda:{}".format(gpu_device) if gpu_device >=0 else 'cpu')

	# In order to use a GPU uncomment the following line. The number is the device index of the used GPU
	# Here, the GPU with the index 0 is used.
	# device = th.device("cuda:0")

	# load the image data and normalize to [0, 1]
	# add mask to loss function
	fixed_image = al.utils.image.create_tensor_image_from_itk_image(sitk.GetImageFromArray(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)), dtype=th.float32, device='cpu')#device,al.read_image_as_tensor("./practice_reg/1.png", dtype=dtype, device=device)#th.tensor(img1,device='cuda',dtype=dtype)#
	moving_image = al.utils.image.create_tensor_image_from_itk_image(sitk.GetImageFromArray(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)), dtype=th.float32, device='cpu')#device,al.read_image_as_tensor("./practice_reg/2.png", dtype=dtype, device=device)#th.tensor(img2,device='cuda',dtype=dtype)#

	fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

	# convert intensities so that the object intensities are 1 and the background 0. This is important in order to
	# calculate the center of mass of the object
	fixed_image.image = 1 - fixed_image.image
	moving_image.image = 1 - moving_image.image

	# create pairwise registration object
	registration = al.PairwiseRegistration()#half=half

	transforms=dict(similarity=al.transformation.pairwise.SimilarityTransformation,
				   affine=al.transformation.pairwise.AffineTransformation,
				   rigid=al.transformation.pairwise.RigidTransformation,
				   non_parametric=al.transformation.pairwise.NonParametricTransformation,
				   wendland=al.transformation.pairwise.WendlandKernelTransformation,
				   bspline=al.transformation.pairwise.BsplineTransformation)
	constant_flow=None

	if transform_type in ['similarity', 'affine', 'rigid']:
		transform_opts=dict(opt_cm=opt_cm)
		transform_args=[0]
		sigma,fixed_image_pyramid,moving_image_pyramid=[[]],[fixed_image],[moving_image]
	else:
		transform_opts=dict(diffeomorphic=opt_cm, device=('cuda:{}'.format(gpu_device) if gpu_device>=0 else 'cpu'))
		transform_args=[moving_image.size]
		if transform_type in ['bspline','wendland']:
			transform_opts['sigma']=sigma
			fixed_image_pyramid = al.create_image_pyramid(fixed_image, pyramid)
			moving_image_pyramid = al.create_image_pyramid(moving_image, pyramid)
		else:
			sigma,fixed_image_pyramid,moving_image_pyramid=[[]],[fixed_image],[moving_image]
		if transform_type=='bspline':
			transform_opts['order']=order
		if transform_type=='wendland':
			transform_opts['cp_scale']=order

	# transform_opts['half']=half

	for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):
		mov_im_level=mov_im_level.to(dtype=th.float32, device=device)
		fix_im_level=fix_im_level.to(dtype=th.float32, device=device)
		# choose the affine transformation model
		if transform_type == 'non_parametric':
			transform_args[0]=mov_im_level.size
		elif transform_type in ['bspline','wendland']:
			# for bspline, sigma must be positive tuple of ints
			# for bspline, smaller sigma tuple means less loss of
			# microarchitectural details
			transform_args[0]=mov_im_level.size
			# transform_opts['sigma'] = sigma[level]
			transform_opts['sigma'] = sigma[level]#(1, 1)
		else:
			transform_args[0]=mov_im_level

		transformation = transforms[transform_type](*transform_args,**transform_opts)

		# if half:
		# 	mov_im_level=mov_im_level.to(dtype=th.float16, device=device)

		# transformation=transformation.to(dtype=th.float32, device=device)# dtype=th.float32,  if not half else th.float16

		if level > 0 and transform_type in ['bspline','wendland']:
			print(interpolation)
			constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
																		  mov_im_level.size,
																		  interpolation=interpolation)
			transformation.set_constant_flow(constant_flow)

		#
		if transform_type in ['similarity', 'affine', 'rigid']:
			# initialize the translation with the center of mass of the fixed image
			transformation.init_translation(fix_im_level)
		# if half:
		# 	fix_im_level=fix_im_level.to(dtype=th.float16, device=device)
		# 	transformation._dtype=th.float16
		# 	transformation._device=device



		optimizer = th.optim.Adam(transformation.parameters(), lr=lr[level], amsgrad=True)
		# opt_level = "O2" if half else "O1"
		# transformation, optimizer = amp.initialize(transformation, optimizer, opt_level=opt_level)

		registration.set_transformation(transformation)

		loss_fns=dict(mse=al.loss.pairwise.MSE,
					ncc=al.loss.pairwise.NCC,
					lcc=al.loss.pairwise.LCC,
					mi=al.loss.pairwise.MI,
					mgf=al.loss.pairwise.NGF,
					ssim=al.loss.pairwise.SSIM)



		# choose the Mean Squared Error as image loss
		image_loss = loss_fns[loss_fn](fix_im_level, mov_im_level)

		registration.set_image_loss([image_loss])

		if transform_type in ['non_parametric','wendland','bspline']:
			regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
			regulariser.set_weight(regularisation_weight[level])
			registration.set_regulariser_displacement([regulariser])

		# choose the Adam optimizer to minimize the objective

		registration.set_optimizer(optimizer)
		registration.set_number_of_iterations(iterations)

		# start the registration
		registration.start()

		if transform_type in ['bspline','wendland']:
			constant_flow = transformation.get_flow()

	# set the intensities back to the original for the visualisation
	fixed_image.image = 1 - fixed_image.image
	moving_image.image = 1 - moving_image.image

	# warp the moving image with the final transformation result
	displacement = transformation.get_displacement()
	# warped_image = al.transformation.utils.warp_image(moving_image, displacement)

	end = time.perf_counter()

	print("=================================================================")

	print("Registration done in:", end - start, "s")
	if transform_type in ['similarity', 'affine', 'rigid']:
		print("Result parameters:")
		transformation.print()

	# # plot the results
	# plt.subplot(131)
	# plt.imshow(fixed_image.numpy(), cmap='gray')
	# plt.title('Fixed Image')
	#
	# plt.subplot(132)
	# plt.imshow(moving_image.numpy(), cmap='gray')
	# plt.title('Moving Image')
	#
	# plt.subplot(133)
	# plt.imshow(warped_image.numpy(), cmap='gray')
	# plt.title('Warped Moving Image')

	if transform_type in ['similarity', 'affine', 'rigid']:
		transformation_param = transformation._phi_z
	elif transform_type == 'non_parametric':
		transformation_param = transformation.trans_parameters
	elif transform_type == 'bspline' or transform_type == 'wendland':
		transformation_param = transformation._kernel
	else:
		pass
	return displacement, moving_image, transformation_param, registration.loss#.data.item()

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
	if gpu_device>=0:
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

	if gpu_device>=0:
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

def match_image_size(img1,img2,black_background=False):
	white=int(black_background==False)
	fill_color=(np.array([255,255,255])*white).astype(int).tolist()
	dh=int(np.abs((img1.shape[0]-img2.shape[0])))
	if img1.shape[0]>img2.shape[0]:
		img2 = cv2.copyMakeBorder(img2, dh//2+dh%2, dh//2, 0, 0, cv2.BORDER_CONSTANT, value=fill_color)
	elif img1.shape[0]<img2.shape[0]:
		img1 = cv2.copyMakeBorder(img1, dh//2+dh%2, dh//2, 0, 0, cv2.BORDER_CONSTANT, value=fill_color)
	dw=int(np.abs((img1.shape[1]-img2.shape[1])))
	if img1.shape[1]>img2.shape[1]:
		img2 = cv2.copyMakeBorder(img2, 0, 0, dw//2+dw%2, dw//2, cv2.BORDER_CONSTANT, value=fill_color)
	elif img1.shape[1]<img2.shape[1]:
		img1 = cv2.copyMakeBorder(img1, 0, 0, dw//2+dw%2, dw//2, cv2.BORDER_CONSTANT, value=fill_color)
	return img1, img2, dw//2+dw%2, dh//2+dh%2
# add moments, first image should have 1+ corresponding segments / 2 sections
# while first has fewer sections
# add mask !!!
def register_images_(im1_fname='A.npy',
						im2_fname='B.npy',
						connectivity=8,
						apply_watershed=False,
						overwrite=False,
						min_object_size=50000,
						clip=True,
						fix_rotation=True,
						mult_factor=2.0,
						scaling_factor=4.,
						output_dir='practice_reg_results/images',
						properties=['area',
									'convex_area',
									'eccentricity',
									'major_axis_length',
									'minor_axis_length',
									'inertia_tensor',
									'inertia_tensor_eigvals',
									'perimeter',
									'orientation'],
						gpu_device=0,
						max_rotation_vertical_px=0,
						loss_fn='mse',
						lr=[0.01]*3,
						transform_type='rigid',
						iterations=1000,
						no_segment_analysis=False,
						black_background=False,
						verbose=False,
						opt_cm=True,
						sigma=[[11,11],[11,11],[3,3]],
						order=2,
						pyramid=[[4,4],[2,2]],
						interpolation='bicubic',
						half=False,
						regularisation_weight=[1,5,50],
						points1='',
						points2=''):

	print("Loading images.")

	_, file_ext = os.path.splitext(im1_fname)

	# TODO: don't rerun splitext, see variable file_ext
	os.makedirs(output_dir, exist_ok=True)
	# img_splitext1 and img_splitext2 are pairs, (root, ext)
	img_splitext1 = os.path.splitext(os.path.basename(im1_fname))
	img_out1 = os.path.join(
		output_dir,
		f"{img_splitext1[0]}_registered{img_splitext1[1]}"
	)
	img_splitext2 = os.path.splitext(os.path.basename(im2_fname))
	img_out2 = os.path.join(
		output_dir,
		f"{img_splitext2[0]}_registered{img_splitext2[1]}"
	)

	if file_ext=='.npy':
		im1=np.load(im1_fname)
		im2=np.load(im2_fname)
	else:
		im1=cv2.imread(im1_fname)
		im2=cv2.imread(im2_fname)

	tre=-1
	if not no_segment_analysis:

		if max_rotation_vertical_px:
			scaling_factor=((sum(im1.shape[:2])+sum(im2.shape[:2]))/4.)/float(max_rotation_vertical_px)
			print("New scaling factor: {}".format(scaling_factor))

		if fix_rotation:
			print("Adjusting 90 degree rotations.")

			im2=correct_rotation(im1, im2, scaling_factor=scaling_factor, gpu_device=gpu_device)

		if gpu_device>=0:
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

		if file_ext=='.npy':

			del im1, im2

			im1=np.load(im1_fname,mmap_mode='r+')
			im2=np.load(im2_fname,mmap_mode='r+')

		print("Performing alignments on pairs of sections.")

		for idx,(i,j) in enumerate(sections):
			if os.path.exists(img_out1)==False or os.path.exists(img_out1)==False or overwrite:

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

					img1,img2=match_image_size(img1,img2,black_background=black_background)[:2]

					print("[{}/{}] - Begin alignment of sections.".format(idx+1,N))

					with (suppress_stdout() if not verbose else contextlib.suppress()):
						new_img=displace_image(img2,affine_register(img1, img2, gpu_device=gpu_device, lr=lr, loss_fn=loss_fn, transform_type=transform_type, iterations=iterations, opt_cm=opt_cm, sigma=sigma, order=order, pyramid=pyramid,interpolation=interpolation, half=half, regularisation_weight=regularisation_weight)[0],gpu_device=gpu_device, dtype=th.float32 if not half else th.half) # new tri, output he as well

					if gpu_device>=0:
						th.cuda.empty_cache()

					print("[{}/{}] - Writing registered sections to file.".format(idx+1,N))

					cv2.imwrite(img_out1,cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
					cv2.imwrite(img_out2,cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))

				except Exception as e:
					print(str(e))

	else:

		im1,im2,dw,dh=match_image_size(im1,im2,black_background=black_background)

		print("Performing registration.")

		with (suppress_stdout() if not verbose else contextlib.suppress()):
			displacement,m_im=affine_register(im1, im2, gpu_device=gpu_device, lr=lr, loss_fn=loss_fn, transform_type=transform_type, iterations=iterations, opt_cm=opt_cm, sigma=sigma, order=order, pyramid=pyramid,interpolation=interpolation, half=half, regularisation_weight=regularisation_weight)[:2]
		new_img=displace_image(im2,displacement,gpu_device=gpu_device, dtype=th.float32 if not half else th.half) # new tri, output he as well

		if points1 and points2 and os.path.exists(points1) and os.path.exists(points2):
			displacement = al.transformation.utils.unit_displacement_to_displacement(displacement)  # unit measures to image domain measures
			displacement = al.create_displacement_image_from_image(displacement, m_im)
			points1=pd.read_csv(points1,index_col=0).values
			points2=pd.read_csv(points2,index_col=0).values
			points2[:,0]+=dw
			points2[:,1]+=dh
			tre=al.utils.points.Points.TRE(points1,al.utils.points.Points.transform(points2,displacement))
		else:
			tre=-1
		if gpu_device>=0:
			th.cuda.empty_cache()

		print("Writing registered section to file.")

		cv2.imwrite(img_out1, cv2.cvtColor(im1,cv2.COLOR_BGR2RGB))
		cv2.imwrite(img_out2, cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))
	return tre

class Commands(object):
	def __init__(self):
		pass

	def apply_drop2_transform(self,
								source_image='A.png',
								ref_image='B.png',
								dx='warp_field_x.nii.gz',
								dy='warp_field_y.nii.gz',
								gpu_device=-1,
								output_file=''):# list of -1s and 1s
		import nibabel
		assert source_image.split('.')[-1]=='png' and ref_image.split('.')[-1]=='png'
		assert os.path.exists(source_image)
		assert os.path.exists(ref_image)
		flip_pattern=[1,1,1,1]
		flip_xy=False
		source_img=cv2.imread(source_image)#cv2.cvtColor(,cv2.COLOR_BGR2RGB)
		ref_img=cv2.imread(ref_image)
		source_img=cv2.resize(source_img,ref_img.shape[-2::-1])
		dx,dy=nibabel.load(dx).get_fdata(),nibabel.load(dy).get_fdata()
		displacement=th.tensor(np.concatenate([dx[::flip_pattern[0],::flip_pattern[1]],dy[::flip_pattern[2],::flip_pattern[3]]][::(-1 if flip_xy else 1)],-1)[::1,::1,...].copy()).unsqueeze(0).float().permute(0,2,1,3)
		for dim in range(displacement.shape[-1]):
			displacement[...,dim]=2.0*displacement[...,dim]/float(displacement.shape[-dim - 2] - 1)
		if gpu_device >= 0:
			displacement=displacement.cuda()
		new_img=displace_image(source_img, displacement, gpu_device)
		cv2.imwrite(source_image.replace('.png','_warped.png') if not output_file else output_file,new_img)

	def compress_image(self,
							im='A.png',
							compression_factor=2.,
							im_out='A.compressed.png'):
		fx=fy=1/compression_factor
		cv2.imwrite(im_out,cv2.resize(cv2.imread(im),None,fx=fx,fy=fy,interpolation=cv2.INTER_CUBIC))

	def register_images(self,
							im1='A.npy',
							im2='B.npy',
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
							gpu_device=0,
							loss_fn='mse',
							lr=0.01,
							transform_type='rigid',
							iterations=1000,
							no_segment_analysis=False,
							black_background=False,
							verbose=False,
							opt_cm=True,
							sigma=[[11,11],[11,11],[3,3]],
							order=2,
							pyramid=[[4,4],[2,2]],
							interpolation='bicubic',
							half=False,
							regularisation_weight=[1,5,50],
							points1='',
							points2='',
							tre_dictionary='results.p'):
		tre=register_images_(im1_fname=im1,
							im2_fname=im2,
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
							max_rotation_vertical_px=max_rotation_vertical_px,
							loss_fn=loss_fn,
							lr=lr,
							transform_type=transform_type,
							iterations=iterations,
							no_segment_analysis=no_segment_analysis,
							black_background=black_background,
							verbose=verbose,
							opt_cm=opt_cm,
							sigma=sigma,
							order=order,
							pyramid=pyramid,
							interpolation=interpolation,
							half=half,
							regularisation_weight=regularisation_weight,
							points1=points1,
							points2=points2)
		if os.path.exists(tre_dictionary):
			tre_dict=pickle.load(open(tre_dictionary,'rb'))
		else:
			tre_dict=dict()
		tre_dict[transform_type,loss_fn]=tre
		pickle.dump(tre_dict,open(tre_dictionary,'wb'))

def main():
	fire.Fire(Commands)

if __name__=='__main__':
	main()
