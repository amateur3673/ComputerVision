import numpy as np
import cv2
def calc_mag_ori(img,shape=(64,128)):
    '''
    Calculate the magnitude and orientation for the img
    Paramters:
    img: The image, can be 2D array for grayscale image or 3D array for RGB image
    shape: the desired shape
    '''
    #First resize the image into the desired shape
    img=cv2.resize(img,shape)
    #Define the kernel for convolution operator
    xkernel=np.array([[-1,0,1]])
    ykernel=np.array([[-1],[0],[1]])
    #convolve image with these kernel and get magnitude and orientation
    dx=cv2.filter2D(img,cv2.CV_32F,xkernel)
    dy=cv2.filter2D(img,cv2.CV_32F,ykernel)
    mag=np.sqrt(dx**2+dy**2) #calculate the magnitude
    dx[np.where(np.abs(dx)<1e-6)]=1e-6 #Avoid zero in dy/dx
    dy[np.where(np.abs(dy)<1e-6)]=1e-6
    ori=np.rad2deg(np.arctan(dy/dx)) #orientation
    ori[np.where(ori<0.0)]+=180 #convert from -90,0 to 90,180
    if(len(img.shape)==2):
        #In grayscale image, we return the mag and ori
        return mag,ori
    else:
        #In RGB image, for each pixel, we choose the channel with the highest magnitude
        channel_idx=np.argmax(mag,axis=2) #Index of channel of the highest magnitude for each image pixel 
        cc,rr=np.meshgrid(np.arange(shape[0]),np.arange(shape[1])) #Get the index of row and column
        mag=mag[rr,cc,channel_idx] #Retrive orientation
        ori=ori[rr,cc,channel_idx] #Retrive orientation
        return mag,ori
def create_cell_histogram(cell_mag,cell_ori,nbins=9):
    '''
    Create the histogram for a cell
    cell_mag: the magnitude matrix of a cell
    cell_ori: the orientation matrix of a cell
    nbins: number of bins used in each cell
    '''
    histogram=np.zeros((nbins,))
    bin_length=180//nbins
    for h in range(cell_ori.shape[0]):
        for w in range(cell_ori.shape[1]):
            orientation=cell_ori[h,w]
            prev_bin_pos=orientation//20 #previous bin position
            next_bin_pos=(prev_bin_pos+1)%nbins #next bin position
            histogram[prev_bin_pos%nbins]+=((prev_bin_pos+1)*bin_length-orientation)*cell_mag[h,w]/bin_length
            histogram[next_bin_pos]+=(orientation-prev_bin_pos*bin_length)*cell_mag[h,w]/bin_length
    return histogram
def create_histogram(mag,ori,cell_shape=(8,8),nbins=9):
    '''
    Create the histogram for all the mangnitude image
    Parameters:
    mag: the magnitude of each pixel
    ori: the orientation of each pixel
    shape: the image patch used for orientation voting
    nbins: number of bins used in each cell
    '''
    cellW=mag.shape[1]//cell_shape[1] #number of cells in W direction
    cellH=mag.shape[0]//cell_shape[0] #number of cells in H direction
    img_his=np.zeros((cellH,cellW,nbins)) #The histogram for the whole image
    for h in range(cellH):
        for w in range(cellW):
            cell_mag=mag[h*cellH:(h+1)*cellH,w*cellW:(w+1)*cellW] #extract the magnitude and orientation in current cell
            cell_ori=ori[h*cellH:(h+1)*cellH,w*cellW:(w+1)*cellW] #extract the magnitude and orientation in current cell
            histogram=create_cell_histogram(cell_mag,cell_ori,nbins)
            img_his[h,w,:]=histogram
    return img_his
def normalize_block(img_histogram,block_shape=(2,2),stride=1,epsilon=1e-5):
    '''
    Normalize in each block. Parameters:
    img_histogram: histogram of each cell, 3D numpy array
    block_shape: shape of block for normalize
    stride: stride (we use overlap)
    epsilon: threshold (to advoid zero when normalize)
    '''
    norm_hist_width=(img_histogram.shape[1]-block_shape[1])//stride+1 #Number of block in width direction
    norm_hist_height=(img_histogram.shape[0]-block_shape[0])//stride+1 #number of block in height direction
    norm_img=np.zeros((norm_hist_height,norm_hist_width,block_shape[0]*block_shape[1]*img_histogram.shape[2]))
    for h in range(norm_hist_height):
        for w in range(norm_hist_width):
            block=img_histogram[h*stride:h*stride+block_shape[0],w*stride:w*stride+block_shape[1],:].flatten()
            normed_block=block/(np.sqrt(np.sum(block**2+epsilon**2)))
            norm_img[h,w,:]=normed_block
    return norm_img.flatten()
def main(image='Image1.png',RGB=True):
    try:
        if(RGB):
            img=cv2.imread('Images/'+image,1) #Read RGB
        else:
            img=cv2.imread('Images/'+image,0) #Grayscale
        mag,ori=calc_mag_ori(img,shape=(64,128))
        ori=ori.astype(int) #Use int type for simplicity
        img_histogram=create_histogram(mag,ori)
        norm_img=normalize_block(img_histogram)
        return norm_img
    except FileNotFoundError:
        print('Cannot open file')
        print('File not found error')
if __name__=='__main__':
    norm_img=main(image='Image1.png')