import numpy as np
import cv2
import matplotlib.pyplot as plt
def generate_octave_blur(img,nlayers,sigma):
    '''
    Generate the Gaussian layers in an Octave by Gaussian Blur
    Parameters:
    img: the image of the last octave that standard deviation is twice as in the begin of the sigma
    nlayers: number of layers in an octave
    sigma: standard deviation
    '''
    size=2*int(3*sigma)+1 #size of the kernel filter
    k=2**(1/nlayers)
    sig=np.zeros((nlayers+3,)) #generate the sigma for each layer
    sig[0]=sigma
    for i in range(1,nlayers+3):
        sig_prev=np.power(k,i-1)*sigma
        sig_total=k*sig_prev
        sig[i]=np.sqrt(sig_total*sig_total-sig_prev*sig_prev)
    octave_img=np.zeros((img.shape[0],img.shape[1],nlayers+3)).astype(np.uint8) #initialize the octave_img for the whole octave
    octave_img[:,:,0]=img
    for i in range(1,nlayers+3):
        octave_img[:,:,i]=cv2.GaussianBlur(img,(size,size),sig[i])
    return octave_img,sig
def generate_pyramid_blur(img,nlayers,noctaves,sigma):
    '''
    Generate the pyramid blur for the image. Parameters:
    img: image
    nlayers: number of layers in an octave
    noctaves: number of octaves
    sigma: standard deviation of Gaussian Filter
    '''
    pyr=[] #initilize the list pyramid
    for i in range(noctaves):
        octave_img,_=generate_octave_blur(img,nlayers,sigma)
        pyr.append(octave_img)
        img=cv2.resize(octave_img[:,:,nlayers+1],(img.shape[0]//2,img.shape[1]//2),cv2.INTER_NEAREST)
    return pyr

def generate_DoG_octave(octave_img):
    '''
    For generating the DoG image in each octave
    Parameters:
    octave_img: the image in the octave (numpy array)
    '''
    nimages=octave_img.shape[2]
    DoG_img=np.zeros((octave_img.shape[0],octave_img.shape[1],nimages-1)).astype(np.uint8)
    for i in range(nimages-1):
        DoG_img[:,:,i]=cv2.subtract(octave_img[:,:,i+1],octave_img[:,:,i])
    return DoG_img
def generate_DoG_pyramid(pyr):
    '''
    Generate the pyramid for DoG
    Parameters:
    pyr: the pyramid (list)
    '''
    DoG_pyr=[]
    for octave in pyr:
        DoG_pyr.append(generate_DoG_octave(octave_img=octave))
    return DoG_pyr

def convert_range(DoG_pyr):
    '''
    convert the DoG_pyr from range 0-255 to range [0,1]
    Parameters:
    DoG_pyr: pyramid, list of numpy array in range 0-255
    '''
    new_DoG_pyr=[]
    for DoG_octave in DoG_pyr:
        new_DoG_pyr.append(DoG_octave/255)
    return new_DoG_pyr

def check_for_scale_extrema(previous_slice,current_slice,next_slice):
    '''
    Check for a point to be the extrema.
    Parameters:
    previous_slice: the neighbors in previous scale, (3,3) numpy array
    current_slice: the neighbors in current scale, (3,3) numpy array
    next_slice: the neighbors in next scale, (3,3) numpy array
    '''
    point=current_slice[1,1]
    if(point>np.max(current_slice[0,:]) and point>current_slice[1,0] and point>current_slice[1,2] and point>np.max(current_slice[2,:])):
        if(point>np.max(previous_slice) and point >np.max(next_slice)):return True
        else: return False
    elif(point<np.min(current_slice[0,:]) and point<current_slice[1,0] and point<current_slice[1,2] and point<np.min(current_slice[2,:])):
        if(point<np.min(previous_slice) and point < np.min(next_slice)):return True
        else: return False
    return False
def find_potential_extrema_octave(DoG_img,border=10):
    '''
    Find the potential extrema in an DoG octave
    Parameters:
    octave_idx: the index of the octave
    DoG_img: the DoG img in the octave, 3D numpy array
    border: the border to the left, right, top and bottom, which we define the scope for searching the extrema
    '''
    octave_keypoints=[] #initialize the list of keypoints
    nlayers=DoG_img.shape[2]
    for layer_idx in range(1,nlayers-1):
        for y in range(border,DoG_img.shape[0]-border):
            for x in range(border,DoG_img.shape[1]-border):
                current_slice=DoG_img[y-1:y+2,x-1:x+2,layer_idx] #Extract the slice of neighbors in the current layer
                previous_slice=DoG_img[y-1:y+2,x-1:x+2,layer_idx-1] #Extract the corresponding neighbors in the previous layer
                next_slice=DoG_img[y-1:y+2,x-1:x+2,layer_idx+1] #Extract the corresponding neighbors in the next layer
                if(check_for_scale_extrema(previous_slice,current_slice,next_slice)):octave_keypoints.append([x,y,layer_idx])
    return np.array(octave_keypoints)
def find_potential_extrema_pyr(DoG_pyr,border=10):
    '''
    Find the potential extrema in the pyramid
    Parameters:
    DoG_pyr: DoG pyramid, list of octave_img
    border
    '''
    keypoints=[]
    for octave in DoG_pyr:
        keypoints.append(find_potential_extrema_octave(octave,border))
    return keypoints

def calculate_derivatives(D,x,y,s):
    '''
    We calculate the derivatives for localizing the keypoints
    '''
    dx=(D[y,x+1,s]-D[y,x-1,s])/2
    dy=(D[y+1,x,s]-D[y-1,x,s])/2
    ds=(D[y,x,s+1]-D[y,x,s-1])/2
    dxx=(D[y,x+1,s]-2*D[y,x,s]+D[y,x-1,s])
    dyy=(D[y+1,x,s]-2*D[y,x,s]+D[y-1,x,s])
    dss=(D[y,x,s+1]-2*D[y,x,s]+D[y,x,s-1])
    dxy=((D[y+1,x+1,s]-D[y+1,x-1,s])-(D[y-1,x+1,s]-D[y-1,x-1,s]))/4
    dxs=((D[y,x+1,s+1]-D[y,x-1,s+1])-(D[y,x+1,s-1]-D[y,x-1,s-1]))/4
    dys=((D[y+1,x,s+1]-D[y-1,x,s+1])-(D[y+1,x,s-1]-D[y-1,x,s-1]))/4
    first_diff=np.array([[dx],[dy],[ds]])
    second_diff=np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])
    return first_diff,second_diff
def relocalize_keypoints_octave(DoG_octave_keypoint,DoG_octave,contrast_thr=0.03,ratio_thr=10):
    '''
    Relocalize keypoints in an octave, using the contrast threshold and edge respond
    Parameters: DoG_octave: 3D numpy array
    '''
    octave_keypoints=[] #Reinitialize the keypoints
    for keypoint in DoG_octave_keypoint:
        x,y,s=keypoint[0],keypoint[1],keypoint[2]
        first_diff,second_diff=calculate_derivatives(DoG_octave,x,y,s)
        x_hat = -np.linalg.pinv(second_diff).dot(first_diff) #calculate the x_hat in the paper
        contrast=DoG_octave[y,x,s]+0.5*first_diff.T.dot(x_hat) #Calculate the D(x_hat) in the paper
        if(contrast[0,0]<contrast_thr):continue #Remove low contrast
        else:
            #Remove the edge response
            Hessian=second_diff[:2,:2] #Retrieve the Hessian matrix
            Trace_Hess=Hessian[0,0]+Hessian[1,1] #Trace of the Hessian matrix
            det_Hess=max(Hessian[0,0]*Hessian[1,1]-Hessian[0,1]*Hessian[1,0],1e-6) #determinant of Hessian matrix
            if(Trace_Hess**2/det_Hess>=ratio_thr):continue #eliminate the keypoint less than ratio_thr
            if(x_hat[0,0]>0.5 or x_hat[1,0]>0.5 or x_hat[2,0]>0.5):
                new_keypoint=np.array([int(round(keypoint[0]+x_hat[0,0])),int(round(keypoint[1]+x_hat[1,0])),int(round(keypoint[2]+x_hat[2,0]))])
                octave_keypoints.append(new_keypoint)
            else:
                octave_keypoints.append(keypoint)
    return np.array(octave_keypoints)
def relocalize_keypoints_pyramid(pyr_keypoints,DoG_pyr,contrast_thr=0.03,ratio_thr=10):
    '''
    Relocalize keypoints for the full pyramid
    '''
    new_pyr_keypoints=[]
    for i in range(len(pyr_keypoints)):
        new_pyr_keypoints.append(relocalize_keypoints_octave(pyr_keypoints[i],DoG_pyr[i],contrast_thr,ratio_thr))
    return new_pyr_keypoints

def cal_mag_and_orientation(x,y,gauss_img):
    '''
    Calculate the magnitude and orientation for the gauss_img at location x and y
    Parameters:
    x,y: 2 intergers represent the location
    gauss_img: the gaussian image, np.uint8 data type 
    '''
    diff1=int(gauss_img[y,x+1])-int(gauss_img[y,x-1])
    if(np.abs(diff1)<1e-6):diff1=1e-6
    diff2=int(gauss_img[y+1,x])-int(gauss_img[y-1,x])
    if(np.abs(diff2)<1e-6):diff2=1e-6
    mag=np.sqrt(diff1**2+diff2**2)
    # We calculate for each case to retrive the angle from -180 to 180
    ori=np.rad2deg(np.arctan2(diff2,diff1))
    return mag,ori
def assign_orientation_octave(octave_gauss_img,octave_keypoints,sig,scale_factor=1.5,nbins=36,peak_thr=0.8):
    '''
    Assign the orientation for each keypoint in an octave
    Parameters: octave_gauss_img: the gaussian image in that octave, 3D numpy array
    octave_keypoints: numpy array, represent the keypoints in an octave
    sig: the array sigma we use to blur the image
    scale_factor: the scale factor for Gaussian weighted circular window
    nbins: number of bin in the histogram
    '''
    new_keypoints=[] #Initialize the new keypoints
    for keypoint in octave_keypoints:
        x,y,s=keypoint[0],keypoint[1],keypoint[2] #coordinate in the scale space
        hist=np.zeros((nbins,)) #Initialize the histogram
        sigma=1.5*sig[s] # Sigma of the circular window
        radius=int(round(3*sigma)) #Radious of the circular window
        #Define the start and finish of the circular windows
        x_start=max(0,x-radius);x_end=min(octave_gauss_img.shape[1],x+radius)
        y_start=max(0,y-radius);y_end=min(octave_gauss_img.shape[0],y+radius)
        for i in range(y_start,y_end+1):
            for j in range(x_start,x_end+1):
                mag,ori=cal_mag_and_orientation(j,i,gauss_img=octave_gauss_img[:,:,s])
                weight_factor=1/(2*np.pi*sigma**2)
                weight=weight_factor*np.exp(-((x-j)**2+(y-i)**2)/(2*sigma**2))
                hist_indx=int(round(nbins*ori/360)) #index in the histogram of the current point
                hist[hist_indx%nbins]+=weight*mag #Add the weighted magnitude to the histogram
        peak_value=np.max(hist) #peak in the histogram
        peak_idx=np.where(hist>=peak_thr*peak_value)[0] #Index of peak that greater than 0.8 the peak value
        for idx in peak_idx:
            new_keypoints.append([x,y,s,idx*10]) # add a new orientation to new keypoint list
    return np.array(new_keypoints)
def assign_orientation_pyr(pyr_gauss_img,pyr_keypoints,sig,scale_factor=1.5,nbins=36,peak_thr=0.8):
    '''
    Assign the orientation for all image in the octave
    '''
    new_pyr_keypoints=[]
    for i in range(len(pyr_keypoints)):
        octave_gauss_img=pyr_gauss_img[i]
        octave_keypoints=pyr_keypoints[i]
        new_pyr_keypoints.append(assign_orientation_octave(octave_gauss_img,octave_keypoints,sig,scale_factor,nbins,peak_thr))
    return new_pyr_keypoints

def generate_gaussian_weighted_kernel(sigma,size):
    '''
    Generate a gaussian kernel with standard deviation sigma, extract the corresponding size for the gaussian windows
    '''
    x,y=np.arange(-size//2+1,size//2+1),np.arange(-size//2+1,size//2+1)
    x,y=np.meshgrid(x,y)
    kernel=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernel
def cal_patch_mag_ori(prev_patch_x,prev_patch_y,next_patch_x,next_patch_y):
    '''
    Calculate the magnitude and orientation for the current patch
    prev_patch_x,prev_patch_y: push the current patch back by 1 in x dimension and y in y dimension
    next_patch_x,next_patch_y: push the current patch forward by 1 in x dimension and y by 1 in y dimension
    This is like what we did in section above, except this is for the whole patch image
    '''
    diffx=next_patch_x-prev_patch_x
    diffx[np.where(np.abs(diffx)<1e-6)]=1e-6 #avoid devided by zero
    diffy=next_patch_y-prev_patch_y
    diffy[np.where(np.abs(diffy)<1e-6)]=1e-6 #advoid devided by zero
    mag=np.sqrt(diffx**2+diffy**2)
    ori=np.rad2deg(np.arctan2(diffy,diffx))
    return mag,ori
def generate_subregion_hist(sub_mag,sub_ori,reference_ori,nbins=8):
    '''
    Generate the subregion histogram.
    Parameters:
    sub_mag: the magnitude for the subregion
    sub_ori: the orientation for the subregion
    reference_ori: the orientation of that keypoint
    nbins: number of bins for the histogram
    '''
    bin_width=360/nbins #width of the histogram bins
    hist=np.zeros((nbins,)) #Initialize the histogram
    for i in range(sub_mag.shape[0]):
        for j in range(sub_mag.shape[1]):
            angle=(sub_ori[i,j]-reference_ori)%360 #Angle compared to reference orientation
            hist_idx=int(round(angle/bin_width))%nbins
            hist[hist_idx]+=sub_mag[i,j]
    return hist
def get_octave_descriptor(octave_gauss_img,octave_keypoints,window=16,nbins=8,nregions=4):
    '''
    Retrieve the descriptor for each octave keypoints
    Parameters:
    octave_gauss_img: blur gauss image in the octave
    octave_keypoints: the keypoints of the octave
    window: the scope for calculating the descriptor, here is (16,16)
    nbins: number of bins in each subregion for calculating the histogram
    nregions: number of regions
    '''
    octave_descriptor=[] #Initialize the list of histogram
    sigma=1.5*window #The author said that sigma is one half of the descriptor window
    kernel=generate_gaussian_weighted_kernel(sigma,window) #Generate the kernel for descriptor windows
    subregion_size=window//nregions #Size of each subregion
    for keypoint in octave_keypoints:
        x,y,s=keypoint[0],keypoint[1],keypoint[2]
        #Define the region around keypoint
        x_start=max(x-window//2+1,0);x_end=min(x+window//2+1,octave_gauss_img.shape[1])
        y_start=max(y-window//2+1,0);y_end=min(y+window//2+1,octave_gauss_img.shape[0])
        patch=octave_gauss_img[y_start:y_end,x_start:x_end,s] #patch
        # Define the patch
        prev_patch_x=octave_gauss_img[y_start:y_end,x_start-1:x_end-1,s].astype(np.int16)
        next_patch_x=octave_gauss_img[y_start:y_end,x_start+1:x_end+1,s].astype(np.int16)
        prev_patch_y=octave_gauss_img[y_start-1:y_end-1,x_start:x_end,s].astype(np.int16)
        next_patch_y=octave_gauss_img[y_start+1:y_end+1,x_start:x_end,s].astype(np.int16)
        #Return the mag and ori matrix for the current patch
        mag,ori=cal_patch_mag_ori(prev_patch_x,prev_patch_y,next_patch_x,next_patch_y)
        mag*=kernel #Recalulate the magnitude
        hist=np.zeros((nregions*nregions,nbins)) #Initialize the histogram
        for y_idx in range(nregions):
            for x_idx in range(nregions):
                sub_mag=mag[y_idx*subregion_size:(y_idx+1)*subregion_size,x_idx*subregion_size:(x_idx+1)*subregion_size]
                sub_ori=ori[y_idx*subregion_size:(y_idx+1)*subregion_size,x_idx*subregion_size:(x_idx+1)*subregion_size]
                sub_hist=generate_subregion_hist(sub_mag,sub_ori,keypoint[3],nbins)
                hist[y_idx*nregions+x_idx,:]=sub_hist
        hist=hist.flatten() #Flatten the histogram
        hist=hist/np.linalg.norm(hist) #Normalize by unit
        hist[np.where(hist>0.2)]=0.2 #Set all the histogram value greater than 0.2 equal 0.2
        hist=hist/np.linalg.norm(hist) #Normalize the hist once again
        octave_descriptor.append(hist)
    return np.array(octave_descriptor)
def get_pyramid_descriptor(pyr_gauss_img,pyr_keypoints,window=16,nbins=8,nregions=4):
    '''
    Generate the descriptor for each keypoints in the pyramid
    Parameters: 
    pyr_gauss_img: pyramid of gaussian blur image
    pyr_keypoints: pyramid keypoints
    window: scope for calculating the descriptor
    nbins: number of bins in a region
    nregions: number of region for each dimension
    '''
    pyr_descriptors=[]
    for octave_idx in range(len(pyr_gauss_img)):
        octave_gauss_img=pyr_gauss_img[octave_idx]
        octave_keypoints=pyr_keypoints[octave_idx]
        pyr_descriptors.append(get_octave_descriptor(octave_gauss_img,octave_keypoints,window,nbins,nregions))
    return pyr_descriptors

def retrieve_keypoints_descriptors(pyr_keypoints,pyr_descriptors):
    keypoints=pyr_keypoints[0]
    descriptors=pyr_descriptors[0]
    for idx in range(1,len(pyr_keypoints)):
        keypoints=np.concatenate((keypoints,pyr_keypoints[idx]),axis=0)
        descriptors=np.concatenate((descriptors,pyr_descriptors[idx]),axis=0)
    return keypoints,descriptors

def main(image_path='Images/IMG_0851.JPG'):
    img=cv2.imread(image_path,0)
    img=cv2.resize(img,(800,800))
    _,sig=generate_octave_blur(img,nlayers=3,sigma=1.6)
    pyr=generate_pyramid_blur(img,nlayers=3,noctaves=4,sigma=1.6)
    DoG_pyr=generate_DoG_pyramid(pyr)
    DoG_pyr=convert_range(DoG_pyr)
    keypoints=find_potential_extrema_pyr(DoG_pyr)
    pyr_keypoints=relocalize_keypoints_pyramid(keypoints,DoG_pyr)
    pyr_keypoints=assign_orientation_pyr(pyr,pyr_keypoints,sig)
    pyr_descriptors=get_pyramid_descriptor(pyr,pyr_keypoints)
    keypoints,descriptors=retrieve_keypoints_descriptors(pyr_keypoints,pyr_descriptors)
    return keypoints,descriptors
if __name__=='__main__':
    keypoints,descriptors=main()
    print(keypoints.shape)
    print(descriptors.shape)
