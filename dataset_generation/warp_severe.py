#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os
import random
# import time

from numpy.lib.function_base import meshgrid
# from scipy.ndimage.filters import gaussian_filter

# from skimage.color import rgb2gray
from tqdm import tqdm
from scipy import signal, ndimage, interpolate
from numpy import sin, cos, tan
# from PIL import Image
from imageio import imread



dirname = os.path.dirname(__file__)

def generate_geometry(crack_form=False):
    x,y = np.meshgrid(range(1,513), range(1,513))


    radii_scaling = np.array([
        [0.8, 1.2, 0.5], #major axis
        [1.27, 0.9, 1.5] #minor axis
        ])


    brightness = [(0.08,0.08), (0.55,0.25), (0.45,0.4)] #pixel brightness values arranged to correspond with whichever radii_scaling being used in the i-th loop iteration
    density = [20, 2, 2] #indexing of elipse density is intended to correspond with radii_scaling, just has brightness variable

    radii_scaling = radii_scaling.T  


    # center = np.random.randint(1, 511, [512,2])
    # radius = np.random.rand(512,1) * 1.8 + 1.27
    # ab = np.random.randint(1, 3, [512,2])*4
    img_input = np.zeros((512,512)) + 0.05 #init input image 512x512 greyscale

    # print(center.shape, radius.shape, ab.shape, img_input.shape)

    for ind, j in enumerate(radii_scaling):
        radius = np.random.rand(512,1) * j[0] + j[1]
        center = np.random.randint(1, 511, [512,2])
        ab = np.random.randint(1, 3, [512,2])*4

        how_dense = density[ind]
        for k in np.arange(how_dense):
            for i in np.arange(512):
                rand_int = np.random.randint(0, 511)

                # ir: rotated axis i, jr: rotated axis j
                # notice ir and jr are randomly rotated by applying random indexing rand_int to radii ratio ab
                ir = (x-center[i,0])*cos(ab[rand_int,1]/ab[rand_int,0]) + (y-center[i,1])*sin(ab[rand_int,1]/ab[rand_int,0])
                jr = -(x-center[i,0])*sin(ab[rand_int,1]/ab[rand_int,0]) + (y-center[i,1])*cos(ab[rand_int,1]/ab[rand_int,0])

                # assign randomized brightness values to each pixel in img_input
                img_input[( ir / ab[i,0] ) **2 + ( jr / ab[i,1] ) **2 <= radius[i,0]**2] = np.random.rand() * brightness[ind][0] + brightness[ind][1]

    # plt.figure()
    # plt.imshow(img_input, cmap='gray')
    # plt.savefig('img_input_6.png')
    # plt.show()

    if crack_form == True:
        img_input = crack_formation(img_input)

    img_input = signal.medfilt2d(img_input, kernel_size=7)
    img_input = ndimage.gaussian_filter(img_input, 2)
    # plt.figure()
    # plt.title('image before warping')
    # plt.imshow(img_input, cmap='gray')
    # plt.savefig('raw_img.png')
    # np.savetxt('raw_img.dat', img_input)
    # print('image saved..')
    # plt.show()
    print()
    return img_input

def crack_formation(img_input):

    ##testing parameters. comment parameters define in true use-case below
    # img_input = np.ones((512,512)) #dummy image
    # h = 50
    # L = 300
    # crack_start = 150

    # path = os.path.join(dirname, 'raw_img.dat' )
    # img_input = np.loadtxt(path)

    ##true use-case for randomized triangular crack formation
    h = np.random.randint(1, int(img_input.shape[0]/2)) #define height of triangular crack formation to be very small (h=1) or a max height of one-tenth of the height of image
    L = np.random.randint(int(img_input.shape[0]/5), int(img_input.shape[0]/2.5))
    crack_start = np.random.randint(int(img_input.shape[0]/10),int(img_input.shape[0]/1.5))
    crack_end = crack_start + h
    crack_width_row = int(h/2)

    ##Form the triangular cracks
    #define 3 points that make the triangle
    pt1 = (crack_start, 0)
    pt2 = (crack_width_row, L)
    pt3 = (crack_end, 0)

    # find slopes of lines forming the triangle
    if (pt2[0] - pt1[0] == 0 or pt3[0] - pt2[0] == 0): #check for divide by zero error
        return img_input
    else:
        m1 = (pt2[1] - pt1[1])/(pt2[0] - pt1[0])  #slope between pt2 and pt1
        m2 = (pt3[1] - pt2[1])/(pt3[0] - pt2[0])  #slope between pt3 and pt2

    reversal = np.random.randint(0,2)
    reversal = reversal == 0
    # print(f"generate crack on right side? {reversal} ")
    # print(f"triangle line1 slope: {m1}, triangle line2 slope: {m2}  ")
    for row in np.arange(img_input.shape[0]):
        for col in np.arange(img_input.shape[1]):
            if reversal == 0:
                if (row >= (abs(int(m1*col)) + crack_start)) and (row<= (int(m2*col) +crack_end))  and col <L:
                    img_input[row,col] = 0
            elif reversal ==1:
                if (row >= (abs(int(m1*col)) + crack_start)) and (row<= (int(m2*col) +crack_end))  and col >-L:
                    img_input[row,-col] = 0
                

    return img_input

def gauss2D(x,y,sig1,mu1,sig2,mu2,amp,vo):
    
    xnumerator = (x-mu1)**2
    xdenom = 2*sig1**2
    xterm = xnumerator/xdenom
    
    ynumerator = (y-mu2)**2
    ydenom = 2*sig2**2
    yterm = ynumerator/ydenom
    
    return amp*(np.exp(-(xterm + yterm))) + vo

def warp_image(default=True, img_path='test', crack_form=False, testing=False):

    if crack_form == True and testing == False:
        print("\ngenerating images with crack")
        input_img = generate_geometry(crack_form=True)
    elif crack_form == False and testing == False:
        print("\ngenerating images with no crack")
        input_img = generate_geometry(crack_form=False)
    elif testing == True:
        print("\ntesting warp_image() function using test image .dat file")
        path = os.path.join(dirname, 'images/image_samples/raw_img.dat')
        input_img = np.loadtxt(path)
    
        


        
    # input_img = crack_formation()
    
    
    # print(type(input_img))
    # plt.figure(1)
    # plt.imshow(input_img, cmap='gray')
    # plt.show()

    tx = np.random.rand()*4 - 2 #rand x transation
    ty = np.random.rand()*4 - 2 #rand y translation
    sx = np.random.rand()*0.03 + 0.985 #scale x, stretch/compress x
    sy = 2 - sx #scale y, stretch/compress y
    sh_x = (np.random.rand()-0.5) * 0.1 #sheer x
    sh_y = (np.random.rand()-0.5) * 0.1 #sheer y
    th = random.vonmisesvariate(np.pi, 0)


    translation_matrix = np.array(
        [
            [1  , 0 , 0 ],
            [0  , 1 , 0 ],
            [tx ,  ty , 1 ]
        ]
    )

    scale_matrix = np.array(
        [
            [sx  , 0 , 0 ],
            [0  , sy , 0 ],
            [ 0 , 0  , 1 ]
        ]
    )

    shear_matrix = np.array(
        [
            [1      , sh_y  , 0 ],
            [sh_x   , 1     , 0 ],
            [0      , 0     , 1 ]
        ]
    )

    rotation_matrix = np.array(
        [
            [np.cos(th)  , -np.sin(th) , 0 ],
            [np.cos(th)  , np.sin(th) , 0 ],
            [0 ,  0 , 1 ]
        ]
    )

    # setup the transform matrix
    transform_formulation = translation_matrix @ scale_matrix @ shear_matrix 
    transform_formulation = transform_formulation.T #transpose the matrix so homogenous axis is at bottom row so it works in ndimage.affine_transform()

    # transform_formulation = translation_matrix.dot(scale_matrix).dot(shear_matrix )
    # transform_formulation[2,0] = 0
    # transform_formulation[2,1] = 0
    
    # print('homogenous transformation matrix:')
    # print(f"shape: \n{transform_formulation.shape}")
    # print(transform_formulation)
    
    
    grid_lin = np.linspace(-255.5,255.5, 512)
    x,y = np.meshgrid(grid_lin, grid_lin)
    x = x.astype(float)
    y = y.astype(float)
    
    ##test meshgrid set up
    # print(x.shape, y.shape)
    # print(x[0:10, 0:10])
    # print()
    # print(y[0:10, 0:10])
    # print()
    # print(x[-10:-1, -10:-1])
    # print()
    # print(y[-10:-1, -10:-1])
    # print()
    # print(x[:,253:258])
    # print()
    # print(y[253:258, :])
    # print()
    # plt.figure()
    # plt.imshow(x)
    # plt.title('x gradient before transform')
    # plt.figure()
    # plt.imshow(y)
    # plt.title('y gradient before transform')
    # plt.show()
    


    #compelete the affine transform to generate the actual transformation matrix
    transformx = ndimage.affine_transform(x,np.linalg.inv(transform_formulation))
    transformy = ndimage.affine_transform(y,np.linalg.inv(transform_formulation))
    
    # test affine transform
    # plt.figure()
    # plt.imshow(transformx)
    # plt.title('x gradient AFTER transform')
    # plt.figure()
    # plt.imshow(transformy)
    # plt.title('y gradient AFTER transform')
    # plt.show()
    

    disp_field_x = x-transformx
    disp_field_y = y-transformy
    

    # #testing disp fields    
    # plt.figure()
    # plt.imshow(disp_field_x)
    # plt.colorbar()
    # plt.title('disp field x')
    # plt.figure()
    # plt.imshow(disp_field_y)
    # plt.title('disp field y')
    # plt.colorbar()
    # plt.show()
    
    # disp_field_x = disp_field_x[crop_1:crop_2, crop_1:crop_2] #crop image
    # disp_field_y = disp_field_y[crop_1:crop_2, crop_1:crop_2] #crop image
    
    para = np.random.rand(2,6)*4 - 2
    poisson = np.random.rand()*0.3 + 0.05
    mu1 = (para[0,0]/2 +1)/2
    mu2 = (para[0,2]/2 +1)/2
    sig1 = para[0,1]+2/8 + 0.05
    sig2 = para[0,3]+2/8 + 0.05
    cov1 = sig1**2
    cov2 = sig2**2
    amp = para[0,4]*0.1+0.005*np.sign(para[0,4])
    vo = para[0,5]/2
    
    covariances = [
        [cov1, 0],
        [0, cov2]
    ]

    grid_lin = np.linspace(0,1, 512)
    Xgrid, Ygrid = np.meshgrid(grid_lin, grid_lin)
    # plt.figure()
    # plt.imshow(Xgrid)
    # plt.title('Xgrid')
    # plt.figure()
    # plt.imshow(Ygrid)
    # plt.title('Ygrid')
    # plt.show()

    
    # gausxy = 0.6*mv_norm([mu1, mu2], covariances, (512)).T
    Disp_gaus_1x = 0.6*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)
    Disp_gaus_1y = -0.6*poisson*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)

    # print(Disp_gaus_1x.shape, Disp_gaus_1y.shape)
    
    # plt.figure()
    # plt.imshow(Disp_gaus_1x)
    # plt.title('disp_gaus_1x')
    # plt.figure()
    # plt.plot(Disp_gaus_1x)
    # plt.title('2D gaussian curve U')
    # plt.figure()
    # plt.plot(Disp_gaus_1y)
    # plt.title('2D gaussian curve V')
    # plt.figure()
    # plt.imshow(Disp_gaus_1y)
    # plt.title('disp_gaus_1y')
    # plt.show()


    # para = np.random.rand(2,6)*4 - 2
    # poisson = np.random.rand()*0.3 + 0.05
    mu1 = (para[1,0]/2 +1)/2
    mu2 = (para[1,2]/2 +1)/2
    sig1 = para[1,1]+2/8 + 0.05
    sig2 = para[1,3]+2/8 + 0.05
    cov1 = sig1**2
    cov2 = sig2**2
    amp = para[1,4]*0.1+0.005*np.sign(para[1,4])
    vo = para[1,5]/2
    
    Disp_gaus_2x = -0.6*poisson*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)
    Disp_gaus_2y = 0.6*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)
    
    # print(Disp_gaus_2x.shape)
    # print(mu1, mu2)
    # plt.figure()
    # plt.plot(Disp_gaus_1x)
    # plt.title('2D gaussian curve U')
    # plt.figure()
    # plt.plot(Disp_gaus_1y)
    # plt.title('2D gaussian curve V')
    # plt.figure()
    # plt.plot(Disp_gaus_2x)
    # plt.title('second 2D gaussian curve U')
    # plt.figure()
    # plt.plot(Disp_gaus_2y)
    # plt.title('second 2D gaussian curve V')
    
    # plt.show()

    print()
    disp_field_x = disp_field_x - (Disp_gaus_1x + Disp_gaus_2x)
    disp_field_y = disp_field_y - (Disp_gaus_1y + Disp_gaus_2y)
    

    # print(f"dispx {disp_field_x.shape}\ndispy {disp_field_y.shape}\ndispx_out {disp_field_output[0].shape} \ndispy_out {disp_field_output[1].shape}  ")
    
    # print(Disp_gaus_2x.shape)
    # print(mu1, mu2)

    # print(f"dispx and y max: {np.amax(disp_field_x)}\n {np.amax(disp_field_y)}\n  ")

    # plt.figure()
    # plt.plot(Disp_gaus_2x)
    # plt.figure()
    # plt.plot(Disp_gaus_2y)
    
    # plt.figure()
    # plt.plot(Disp_gaus_1x)
    # plt.figure()
    # plt.plot(Disp_gaus_1y)
    
    # plt.figure()
    # plt.title('disp_field_ux')
    # plt.imshow(disp_field_x[50:450,50:450])
    # plt.colorbar()
    # plt.figure()
    # plt.title('disp_field_uy')
    # plt.imshow(disp_field_y[50:450,50:450])
    # plt.colorbar()
    # plt.show()

    Xgrid, Ygrid = meshgrid(range(0,512), range(0,512)) #create 512x512 meshgrid containing range of values 1:512
    # plt.figure()
    # plt.plot(Xgrid)
    # plt.title('Xgrid')
    # plt.figure()
    # plt.plot(Ygrid)
    # plt.title('Ygrid')
    # # plt.show()


    # crop the distortion fields 
    # crop input image to fit the same size
    crop_1 = 50
    crop_2 = 450
    
    Xgrid_d = Xgrid - disp_field_x
    Ygrid_d = Ygrid - disp_field_y
    Xgrid_d = Xgrid_d[crop_1:crop_2, crop_1:crop_2]
    Ygrid_d = Ygrid_d[crop_1:crop_2, crop_1:crop_2]
    
    Xgrid = Xgrid[crop_1:crop_2, crop_1:crop_2]
    Ygrid = Ygrid[crop_1:crop_2, crop_1:crop_2]

    input_img = input_img[crop_1:crop_2, crop_1:crop_2]

    disp_field_x = disp_field_x[crop_1:crop_2, crop_1:crop_2]
    disp_field_y = disp_field_y[crop_1:crop_2, crop_1:crop_2]

    # plt.figure()
    # plt.imshow(Xgrid)
    # plt.colorbar()
    # plt.title('Xgrid')
    # plt.figure()
    # plt.imshow(Ygrid)
    # plt.colorbar()
    # plt.title('Ygrid')
    # # plt.show()    

    # plt.figure()
    # plt.imshow(Xgrid_d)
    # plt.title('Xgrid_d')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(Ygrid_d)
    # plt.title('Ygrid_d')
    # plt.colorbar()
    # plt.show()
    
    # print(f" {Xgrid_d.shape} {Ygrid_d.shape} ")
    # print(f" {Xgrid.shape} {Ygrid.shape} ")
    # print(f"{input_img[crop_1:crop_2, crop_1:crop_2].shape} ")

    warped_in = np.zeros(input_img.shape)
    for ind, row in enumerate(input_img):
        warped_in[ind] = interpolate.griddata(Xgrid_d[ind,:], row, Xgrid[ind,:], method='cubic')
        

    # interpolate along Ygrid_d not working for some reason. Leaving commented out
    warped = np.zeros(input_img.shape)
    for ind,row in enumerate(warped_in):
        warped[:,ind] = interpolate.griddata(Ygrid_d[:,ind], warped_in[:,ind], Ygrid[:,ind], method='cubic')


    # warped = interpolate.griddata(Ygrid_d[:,0], input_img, Ygrid[:,0], method='cubic')
    # print(f" {Xgrid_d.shape} {Ygrid_d.shape} ")
    
    # print(f'\n{disp_field_x.shape=}, {disp_field_y.shape=}')
    # print(f'{input_img.shape=}\n ')

    # crop the images using a random corner point
    crop_end = (crop_2 - crop_1) - 256
    crop_pt = np.random.randint(0,crop_end) #limit crop pt picking to ((crop_2 - crop_1) - 256) so we ensure the final gt file saved is 256x256 and does not exceed the matrix size after crop_1:crop_2 cropping


    # crop the disp fields accordingly and store both in same list to export as .mat file
    disp_field_x = disp_field_x[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    disp_field_y = disp_field_y[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    disp_field_output = []
    disp_field_output.append(disp_field_x)
    disp_field_output.append(disp_field_y)
    disp_field_output = np.asarray(disp_field_output)
    print()





    ## SEVERE WARPING
    # coeff_sx = -2.15 #0.03
    # coeff_sy = 2 #2
    coeff_shear = 0.5 #0.1
    
    # sx = np.random.rand()*coeff_sx + 0.985 #scale x, stretch/compress x
    sy = 1#coeff_sy - sx #scale y, stretch/compress y
    sh_x = (np.random.rand()-0.5) * coeff_shear #sheer x
    sh_y = (np.random.rand()-0.5) * coeff_shear #sheer y
    th = random.vonmisesvariate(np.pi, 0)

    coeff_sx = -0.05 #0.03
    sy = 1
    sx = np.random.rand()*coeff_sx + 0.985 #scale x, stretch/compress x
    
    scale_matrix = np.array(
        [
            [sx  , 0 , 0 ],
            [0  , sy , 0 ],
            [ 0 , 0  , 1 ]
        ]
    )



    # setup the transform matrix
    transform_formulation = scale_matrix  
    transform_formulation = transform_formulation.T #transpose the matrix so homogenous axis is at bottom row so it works in ndimage.affine_transform()    

    grid_lin = np.linspace(-255.5,255.5, 512)
    x,y = np.meshgrid(grid_lin, grid_lin)
    x = x.astype(float)
    y = y.astype(float)


    #compelete the affine transform to generate the actual transformation matrix
    transformx = ndimage.affine_transform(x,np.linalg.inv(transform_formulation))
    transformy = ndimage.affine_transform(y,np.linalg.inv(transform_formulation))


    disp_field_x = x-transformx
    disp_field_y = y-transformy
    
    
    para = np.random.rand(2,6)*4 - 2
    poisson = np.random.rand()*0.3 + 0.05
    mu1 = (para[0,0]/2 +1)/2
    mu2 = (para[0,2]/2 +1)/2
    sig1 = para[0,1]+2/8 + 0.05
    sig2 = para[0,3]+2/8 + 0.05
    cov1 = sig1**2
    cov2 = sig2**2
    amp = para[0,4]*0.1+0.005*np.sign(para[0,4])
    vo = para[0,5]/2
    
    covariances = [
        [cov1, 0],
        [0, cov2]
    ]

    grid_lin = np.linspace(0,1, 512)
    Xgrid, Ygrid = np.meshgrid(grid_lin, grid_lin)

    
    # gausxy = 0.6*mv_norm([mu1, mu2], covariances, (512)).T
    Disp_gaus_1x = 0.6*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)
    Disp_gaus_1y = -0.6*poisson*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)

    mu1 = (para[1,0]/2 +1)/2
    mu2 = (para[1,2]/2 +1)/2
    sig1 = para[1,1]+2/8 + 0.05
    sig2 = para[1,3]+2/8 + 0.05
    cov1 = sig1**2
    cov2 = sig2**2
    amp = para[1,4]*0.1+0.005*np.sign(para[1,4])
    vo = para[1,5]/2
    
    Disp_gaus_2x = -0.6*poisson*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)
    Disp_gaus_2y = 0.6*gauss2D(Xgrid,Ygrid,sig1,mu1,sig2,mu2,amp,vo)
    
    print()
    # disp_field_x = disp_field_x - (Disp_gaus_1x + Disp_gaus_2x)
    # disp_field_y = disp_field_y - (Disp_gaus_1y + Disp_gaus_2y)
    

    Xgrid, Ygrid = meshgrid(range(0,512), range(0,512)) #create 512x512 meshgrid containing range of values 1:512


    # crop the distortion fields 
    # crop input image to fit the same size


    warped = warped[20:370, 20:370]
    crop_1 = 50 + 20
    crop_2 = 450 - 30
    crop_end = (crop_2 - crop_1) - 256
    crop_pt = np.random.randint(0,crop_end)
    
    Xgrid_d = Xgrid - disp_field_x
    Ygrid_d = Ygrid - disp_field_y


    # input_img = input_img[crop_1:crop_2, crop_1:crop_2]

    # disp_field_x = disp_field_x[crop_1:crop_2, crop_1:crop_2]
    # disp_field_y = disp_field_y[crop_1:crop_2, crop_1:crop_2]


    Xgrid_d = Xgrid_d[crop_1:crop_2, crop_1:crop_2]
    Ygrid_d = Ygrid_d[crop_1:crop_2, crop_1:crop_2]
    
    Xgrid = Xgrid[crop_1:crop_2, crop_1:crop_2]
    Ygrid = Ygrid[crop_1:crop_2, crop_1:crop_2]


    disp_field_x = disp_field_x[crop_1:crop_2, crop_1:crop_2]
    disp_field_y = disp_field_y[crop_1:crop_2, crop_1:crop_2]


    


    # print(f'{warped.shape=}\n {Xgrid_d.shape=}')

    warped_in = np.zeros(warped.shape)
    for ind, row in enumerate(warped):
        warped_in[ind] = interpolate.griddata(Xgrid_d[ind,:], row, Xgrid[ind,:], method='cubic')
        

    # interpolate along Ygrid_d not working for some reason. Leaving commented out
    warped2 = np.zeros(warped.shape)
    for ind,row in enumerate(warped_in):
        warped2[:,ind] = interpolate.griddata(Ygrid_d[:,ind], warped_in[:,ind], Ygrid[:,ind], method='cubic')


    Xgrid_d = Xgrid_d[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    Ygrid_d = Ygrid_d[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    
    Xgrid = Xgrid[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    Ygrid = Ygrid[crop_pt:crop_pt+256, crop_pt:crop_pt+256]


    warped = warped[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    warped2 = warped2[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    input_img = input_img[crop_pt:crop_pt+256, crop_pt:crop_pt+256]

    # crop the disp fields accordingly and store both in same list to export as .mat file
    disp_field_x = disp_field_x[crop_pt:crop_pt+256, crop_pt:crop_pt+256]
    disp_field_y = disp_field_y[crop_pt:crop_pt+256, crop_pt:crop_pt+256]

    disp_field_output2 = []
    disp_field_output2.append(disp_field_x)
    disp_field_output2.append(disp_field_y)
    disp_field_output2 = np.asarray(disp_field_output2)
    print()

    # testing
    # np.savetxt('img_gt.dat', input_img)
    # np.savetxt('img_warped.dat', warped)
    # print(f'\n{crop_pt=} ')
    # print(f'\n{disp_field_output.shape=}')
    # print(f'{input_img.shape=}\n ')
    # plt.figure()
    # plt.title('disp_field_ux out')
    # plt.imshow(disp_field_output[0])
    # plt.colorbar()
    # plt.savefig('disp_field_ux_nocrop.png')
    # plt.figure()
    # plt.title('disp_field_uy out')
    # plt.imshow(disp_field_output[1])
    # plt.colorbar()
    # plt.savefig('disp_field_uy_nocrop.png')
    # plt.show()    

    # TESTING CODE
    # # save gt and warped images without figure window and axes
    # w,h = (1,1) #1x1" window
    # dpi = 255 #dots per inch resolution

    # fig1 = plt.figure(frameon=False)
    # fig1.set_size_inches(w,h)
    # ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
    # ax1.set_axis_off()
    # fig1.add_axes(ax1)
    
    # # plt.figure()
    # plt.imshow(input_img, cmap='gray')
    # # plt.title('image before warping')
    # plt.savefig('img_gt.png', dpi=dpi)
    # # plt.savefig('img_gt.png')
    # # plt.show()
    # print("saved image gt")


    # fig2 = plt.figure(frameon=False)
    # fig2.set_size_inches(w,h)
    # ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
    # ax2.set_axis_off()
    # fig2.add_axes(ax2)
    
    # # plt.figure()
    # # plt.title('image after warping')
    # plt.imshow(warped, cmap='gray')
    # plt.savefig('img_warped.png', dpi=dpi)
    # # plt.savefig('img_warped.png')
    # # plt.show()
    
    # fig3 = plt.figure(frameon=False)
    # fig3.set_size_inches(w,h)
    # ax3 = plt.Axes(fig3, [0., 0., 1., 1.])
    # ax3.set_axis_off()
    # fig3.add_axes(ax3)
    
    # # plt.figure()
    # # plt.title('image after warping')
    # plt.imshow(warped2, cmap='gray')
    # plt.savefig('img_2warped.png', dpi=dpi)
    # # plt.savefig('img_warped.png')
    # # plt.show()
    
    
    # print("saved image warped")
    


    return input_img, warped, warped2, disp_field_output, disp_field_output2



def main(testing=False):
    from scipy.io import savemat
    
    # Let user specify preferances 
    num_samples = int(input('Please enter the desired number N of sample-pairs: '))
    print()
    test_or_train = input('Generate test images or training images? \nEnter ''test'' or ''train'': ')
    print()
    crack_yes = input('\nGenerate cracks into image samples? Enter Y/N: ')
    print()
    
    
    if crack_yes == 'y' or crack_yes == 'Y':
        crack_input = True
    else:
        crack_input = False

    if test_or_train == 'train':
        path_img = os.path.join(dirname , 'images/image_sample_pairs/experiment_samples/imgs_severe')
        path_gt = os.path.join(dirname , 'images/image_sample_pairs/experiment_samples/gts_severe')
    else:
        path_img = os.path.join(dirname , 'images/image_sample_pairs/imgs9/test')
        path_gt = os.path.join(dirname , 'images/image_sample_pairs/gts9/test')
    
    
    start_samples = int(input('start image generation from sample #? : '))
    end_samples = start_samples + num_samples
    print(f'generating {num_samples} sample pairs\n')

    for i in tqdm(np.arange(start_samples,end_samples)):
        print(f"generate sample {i}  ")
        
        
        if testing:
            img_gt, img_warped, img_severe_warp, disp_fields, disp_fields_severe = warp_image(testing=True)
        else:
            img_gt, img_warped, img_severe_warp, disp_fields, disp_fields_severe = warp_image(crack_form=crack_input)
        
        print(f'{img_gt.shape=}, {img_warped.shape=}, {img_severe_warp.shape=}, {disp_fields_severe.shape=} ')

        field_str = os.path.join(path_gt,f"train_image_{i}_.mat")
        savemat(field_str, {'Disp_field_1': disp_fields})
        field_str = os.path.join(path_gt,f"train_image_{i}_severe.mat")
        savemat(field_str, {'Disp_field_1': disp_fields_severe})
        

        # plt.imshow(img_gt, cmap='gray')
        # plt.savefig(os.path.join(path,f"train_image_{i}_1.png"))
        # plt.imshow(img_warped, cmap='gray')
        # plt.savefig(os.path.join(path,f"train_image_{i}_2.png"))
        
        w,h = (1,1)
        dpi=256

        fig1 = plt.figure(frameon=False)
        fig1.set_size_inches(w,h)
        ax1 = plt.Axes(fig1, [0., 0., 1., 1.])
        ax1.set_axis_off()
        fig1.add_axes(ax1)
        # plt.title('image before warping')
        plt.imshow(img_gt, cmap='gray', aspect='auto')
        plt.savefig(os.path.join(path_img,f"train_image_{i}_1.png"), dpi=dpi)

        fig2 = plt.figure(frameon=False)
        fig2.set_size_inches(w,h)
        ax2 = plt.Axes(fig2, [0., 0., 1., 1.])
        ax2.set_axis_off()
        fig2.add_axes(ax2)
        # plt.title('image after warping')
        plt.imshow(img_warped, cmap='gray', aspect='auto')
        plt.savefig(os.path.join(path_img,f"train_image_{i}_2.png"), dpi=dpi)        


        fig3 = plt.figure(frameon=False)
        fig3.set_size_inches(w,h)
        ax3 = plt.Axes(fig3, [0., 0., 1., 1.])
        ax3.set_axis_off()
        fig3.add_axes(ax3)
        # plt.title('image after warping')
        plt.imshow(img_severe_warp, cmap='gray', aspect='auto')
        plt.savefig(os.path.join(path_img,f"train_image_{i}_3.png"), dpi=dpi)        

        plt.close('all')

    print(f'{num_samples} groundtruth & warped images saved...')


if __name__ == '__main__':

    # __ = generate_geometry()
    # __, __, __, __, __ = warp_image(default=False, testing=True) 

    # __,__,__,__,__ = warp_image()
    main()
