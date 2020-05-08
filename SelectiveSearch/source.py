import numpy as np
import cv2
from img_transform import *
###########
# Color similarity
###########

class ColorSimilarity:
    '''
    Measure color similarity between neighboring regions
    '''
    def __init__(self,neighbor_mat_shape):
        '''
        Initialize the similarity matrix between neighbors in the neighbor matrix
        Parameters:
        neighbor_mat: shape of boolean matrix we've computed
        '''
        self.sim_mat=np.zeros((neighbor_mat_shape[0],neighbor_mat_shape[1])).astype(np.float64)
    def compute_similar(self,neighbor_mat,img,list_comp):
        '''
        Compute the color similarity of neighbor components
        Parameters:
        neighbor_mat: the neighbor_matrix we've computed above
        img: image
        '''
        print('compute color similarity ...')
        comp_i,comp_j=np.where(neighbor_mat==True)
        self.hist_arr=self.get_hist_array(neighbor_mat,img,list_comp)
        for idx in range(comp_i.shape[0]):
            self.sim_mat[comp_i[idx],comp_j[idx]]=np.sum(np.minimum(self.hist_arr[comp_i[idx]],self.hist_arr[comp_j[idx]]))
        for idx in range(neighbor_mat.shape[0]):
            self.sim_mat[idx][idx]=0        
    def get_hist_array(self,neighbor_mat,img,list_comp):
        '''
        Form an array of histogram in each region
        '''
        hist_arr=np.zeros((len(list_comp),75))
        for idx in range(hist_arr.shape[0]):
            hist_arr[idx,:]=self.__extract_hist(img,list_comp[idx])
        return hist_arr
    def __extract_hist(self,img,comp,nbins=25):
        '''
        Extract color histogram from component
        Parameters:
        comp: the component we want to extract the histogram
        img: img is an origin image
        nbins: number of bins
        '''
        hist=np.zeros((nbins*3,)) #initialize the histogram
        bin_width=255//nbins
        for pixel in comp:
            for c in range(3):
                if(img[pixel[0],pixel[1],c]>=250):hist[nbins*c+nbins-1]+=1
                else:
                    bin_idx=img[pixel[0],pixel[1],c]//bin_width
                    hist[nbins*c+bin_idx]+=1
        # next use L1 norm to normalize the histogram
        hist=hist/np.sum(abs(hist))
        return hist
    def update(self,pos,neighbor_mat,list_comp):
        '''
        Update after merging the components specified by pos
        Note: pos is a tuple and pos[0]<pos[1]
        '''
        print('Update Color Histogram ...')
        # Compute new histogram
        new_hist=(len(list_comp[pos[0]])*self.hist_arr[pos[0]]+len(list_comp[pos[1]])*self.hist_arr[pos[1]])/(len(list_comp[pos[0]])+len(list_comp[pos[1]]))
        # Extend a row in similarity matrix
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[0],np.zeros((self.sim_mat.shape[1],)),axis=0)
        # Extend a col in similarity matrix
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[1],np.zeros((self.sim_mat.shape[0],)),axis=1)
        # Find the neighboring regions of removed regions
        neigh_re=np.where(np.logical_or(neighbor_mat[pos[0],:],neighbor_mat[pos[1],:])==True)[0]
        # Recompute the new similarity matrix at neigh_re
        for idx in neigh_re:
            self.sim_mat[-1,idx]=np.sum(np.minimum(self.hist_arr[idx],new_hist))
            self.sim_mat[idx,-1]=self.sim_mat[-1,idx]     
        self.hist_arr=np.delete(self.hist_arr,pos[1],axis=0)
        self.hist_arr=np.delete(self.hist_arr,pos[0],axis=0)
        self.hist_arr=np.insert(self.hist_arr,self.hist_arr.shape[0],new_hist,axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=1)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=1)
#########
#Image Transform
#########


##########
#Texture similarity
##########
class TextureSimilarity:
    '''
    Compute texture similarity between regions
    '''
    def __init__(self,neighbor_mat,img):
        self.sim_mat=np.zeros((neighbor_mat.shape[0],neighbor_mat.shape[1]))
        self.setImage(img)

    def setImage(self,img):
        '''
        Compute the gaussian derivative at 8 orientation of the image
        '''
        # No rotation, x direction
        print('Set up the gradiant image ...')
        img_list=[] #Initialize the derivative image
        img_grad=cv2.Scharr(img,cv2.CV_32F,1,0)
        img_list.append(img_grad)
        # No rotation, y direction
        img_grad=cv2.Scharr(img,cv2.CV_32F,0,1)
        img_list.append(img_grad)
        # Rotation, x direction
        rot=rotation_transform(img,center=(img.shape[0]//2,img.shape[1]//2),angle=45.0,interpolation='BILINEAR')
        rot_img=rot._tobox(interpolation='BILINEAR')
        img_grad=cv2.Scharr(rot_img,cv2.CV_32F,1,0)
        img_grad=rotation_transform(img_grad,center=(img_grad.shape[0]//2,img_grad.shape[1]//2),angle=-45.0,interpolation='BILINEAR').rot_img
        offset_0=(img_grad.shape[0]-img.shape[0])//2
        offset_1=(img_grad.shape[1]-img.shape[1])//2
        crop=img_grad[offset_0:img_grad.shape[0]-offset_0,offset_1:img_grad.shape[1]-offset_1,:]
        crop=cv2.resize(crop,(img.shape[1],img.shape[0]))
        img_list.append(crop)
        #Rotation, y direction
        img_grad=cv2.Scharr(rot_img,cv2.CV_32F,0,1)
        img_grad=rotation_transform(img_grad,center=(img_grad.shape[0]//2,img_grad.shape[1]//2),angle=-45.0,interpolation='BILINEAR').rot_img
        crop=img_grad[offset_0:img_grad.shape[0]-offset_0,offset_1:img_grad.shape[1]-offset_1,:]
        crop=cv2.resize(crop,(img.shape[1],img.shape[0]))
        img_list.append(crop)
        print('Thresholding each channel ...')
        self.img_grad=[]
        for image in img_list:
            for c in range(image.shape[2]):
                temp_grad=image[:,:,c]
                self.img_grad.append(cv2.threshold(temp_grad,0,0,cv2.THRESH_TOZERO)[1])
                self.img_grad.append(cv2.threshold(temp_grad,0,0,cv2.THRESH_TOZERO_INV)[1])
    def compute_similar(self,neighbor_mat,list_comp):
        '''
        Compute the similarity matrix of components in the image
        '''
        print('Compute texture similarity ...')
        comp_i,comp_j=np.where(neighbor_mat==True)
        self.hist_arr=self.get_hist_array(neighbor_mat,list_comp)
        for idx in range(comp_i.shape[0]):
            self.sim_mat[comp_i[idx],comp_j[idx]]=np.sum(np.minimum(self.hist_arr[comp_i[idx]],self.hist_arr[comp_j[idx]]))
        for idx in range(neighbor_mat.shape[0]):
            self.sim_mat[idx][idx]=0        
    def get_hist_array(self,neighbor_mat,list_comp,nbins=10):
        '''
        Get the histogram array for all components
        '''
        bin_width=np.array([((np.max(self.img_grad[i])-np.min(self.img_grad[i])+2)/nbins) for i in range(len(self.img_grad))])
        hist_arr=np.zeros((neighbor_mat.shape[0],240)) #Initialize the histogram array
        for idx in range(neighbor_mat.shape[0]):
            hist_arr[idx,:]=self.__extract_hist(list_comp[idx],bin_width)
        return hist_arr
    def __extract_hist(self,comp,bin_width,nbins=10):
        '''
        Compute the histogram of 1 component
        Parameters:
        comp: list of pixel in that component
        nbins: number of bins for each gradiant image channel,set default = 10
        '''
        
        hist=np.zeros((nbins*len(self.img_grad)))
        for pixel in comp:
            for idx in range(len(self.img_grad)):
                bin_idx=int(self.img_grad[idx][pixel[0],pixel[1]]/bin_width[idx])%nbins
                hist[idx*nbins+bin_idx]+=1
        return hist/np.sum(abs(hist))
    def update(self,pos,neighbor_mat,list_comp):
        '''
        Update after merging the components specified by pos
        Note: pos is a tuple and pos[0]<pos[1]
        '''
        print('Update texture similarity ...')
        # Compute new histogram
        new_hist=(len(list_comp[pos[0]])*self.hist_arr[pos[0]]+len(list_comp[pos[1]])*self.hist_arr[pos[1]])/(len(list_comp[pos[0]])+len(list_comp[pos[1]]))
        # Extend a row in similarity matrix
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[0],np.zeros((self.sim_mat.shape[1],)),axis=0)
        # Extend a col in similarity matrix
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[1],np.zeros((self.sim_mat.shape[0],)),axis=1)
        # Find the neighboring regions of removed regions
        neigh_re=np.where(np.logical_or(neighbor_mat[pos[0],:],neighbor_mat[pos[1],:])==True)[0]
        # Recompute the new similarity matrix at neigh_re
        for idx in neigh_re:
            self.sim_mat[-1,idx]=np.sum(np.minimum(self.hist_arr[idx],new_hist))
            self.sim_mat[idx,-1]=self.sim_mat[-1,idx]     
        self.hist_arr=np.delete(self.hist_arr,pos[1],axis=0)
        self.hist_arr=np.delete(self.hist_arr,pos[0],axis=0)
        self.hist_arr=np.insert(self.hist_arr,self.hist_arr.shape[0],new_hist,axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=1)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=1)

################
# Size Similarity
################
class SizeSimilarity:
    '''
    Compute the size similarity between neighboring components
    '''
    def __init__(self,neighbor_mat):
        self.sim_mat=np.zeros((neighbor_mat.shape[0],neighbor_mat.shape[1]))
    def compute_similar(self,neighbor_mat,list_comp,img_area):
        '''
        Compute the size similarity matrix
        '''
        print('Compute size similarity ...')
        comp_i,comp_j=np.where(neighbor_mat==True)
        for idx in range(len(comp_i)):
            self.sim_mat[comp_i[idx],comp_j[idx]]=1-(len(list_comp[comp_i[idx]])+len(list_comp[comp_j[idx]]))/img_area
        for idx in range(neighbor_mat.shape[0]):
            self.sim_mat[idx,idx]=0
    def update(self,pos,neighbor_mat,list_comp,img_area):
        '''
        Update the sim_mat after merging the region specified by pos
        '''
        #Extend the similarity matrix
        print('Update Size similarity ...')
        new_size=len(list_comp[pos[0]])+len(list_comp[pos[1]])
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[0],np.zeros((self.sim_mat.shape[1],)),axis=0)
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[1],np.zeros((self.sim_mat.shape[0],)),axis=1)
        # Find neighboring regions
        neigh_re=np.where(np.logical_or(neighbor_mat[pos[0]],neighbor_mat[pos[1]])==True)[0]
        # Update similarity matrix
        for idx in neigh_re:
            self.sim_mat[-1,idx]=1-(new_size+len(list_comp[idx]))/img_area
            self.sim_mat[idx,-1]=self.sim_mat[-1,idx]
        # Finally remove the merging regions
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=1)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=1)
################
# Fill Similarity
################
class FillSimilarity:
    '''
    Compute fill similarity between neighboring components in the image
    '''
    def __init__(self,neighbor_mat):
        self.sim_mat=np.zeros((neighbor_mat.shape[0],neighbor_mat.shape[1]))
    def compute_similar(self,neighbor_mat,list_comp,img_area):
        '''
        Compute the fill similarity matrix
        '''
        print('Compute fill similarity ...')
        self.bounding=self.comp_bounding(list_comp)
        comp_i,comp_j=np.where(neighbor_mat==True)
        for idx in range(len(comp_i)):
            height=max(self.bounding[comp_i[idx],2],self.bounding[comp_j[idx],2])-min(self.bounding[comp_i[idx],0],self.bounding[comp_j[idx],0])+1
            width=max(self.bounding[comp_i[idx],3],self.bounding[comp_j[idx],3])-min(self.bounding[comp_i[idx],1],self.bounding[comp_j[idx],1])+1
            box_area=height*width
            self.sim_mat[comp_i[idx],comp_j[idx]]=max(0,min(1,1-(box_area-len(list_comp[comp_i[idx]])-len(list_comp[comp_j[idx]]))/img_area))
        for idx in range(neighbor_mat.shape[0]):
            self.sim_mat[idx,idx]=0
    def comp_bounding(self,list_comp):
        '''
        Compute the bounding of 1 component
        '''
        bounding=np.zeros((len(list_comp),4)) # 4 coordinates of the bounding box
        for idx in range(len(list_comp)):
            bounding[idx]=self.get_bounding(list_comp[idx])
        return bounding
    def get_bounding(self,comp):
        '''
        get the bounding of 1 component
        '''
        comp=np.array(comp)
        ymin,xmin,ymax,xmax=np.min(comp[:,0]),np.min(comp[:,1]),np.max(comp[:,0]),np.max(comp[:,1])
        return np.array([ymin,xmin,ymax,xmax])
    def update(self,pos,neighbor_mat,list_comp,img_area):
        '''
        Update merged components specified by pos
        '''
        # Extended the sim_mat
        print('Update Fill similarity ...')
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[0],np.zeros((self.sim_mat.shape[1],)),axis=0)
        self.sim_mat=np.insert(self.sim_mat,self.sim_mat.shape[1],np.zeros((self.sim_mat.shape[0],)),axis=1)
        # Collect neighboring regions
        neigh_re=np.where(np.logical_or(neighbor_mat[pos[0]],neighbor_mat[pos[1]])==True)[0]
        # Calculate the new bounding
        new_bound=self.get_bounding(comp=list_comp[pos[0]]+list_comp[pos[1]])
        self.bounding=np.insert(self.bounding,self.bounding.shape[0],new_bound,axis=0)
        new_size=len(list_comp[pos[0]])+len(list_comp[pos[1]])
        for idx in neigh_re:
            height=max(self.bounding[-1,2],self.bounding[idx,2])-min(self.bounding[-1,0],self.bounding[idx,0])+1
            width=max(self.bounding[-1,3],self.bounding[idx,3])-min(self.bounding[-1,1],self.bounding[idx,1])+1
            box_area=height*width
            self.sim_mat[-1,idx]=max(0,min(1,1-(box_area-new_size-len(list_comp[idx]))/img_area))
        # Remove the merged componets
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=0)
        self.sim_mat=np.delete(self.sim_mat,pos[1],axis=1)
        self.sim_mat=np.delete(self.sim_mat,pos[0],axis=1)


################
# Hierarchical Grouping
################
class Hierarchical:
    '''
    Conduct Hierarchical Grouping Stategy
    '''
    def __init__(self,img,merge=True,file_name='result.txt'):
        # Initialize 4 similarities matrix
        self.setup(img,merge,file_name)
    def setup(self,img,merge=True,file_name='result.txt'):
        print('Set up for Selective Search ...')
        img_comp,self.list_comp=retrieve_components(img,file_name)
        '''self.list_comp=list_comp'''
        self.neighbor_mat=find_neighbors(img_comp,len(self.list_comp))
        if(merge):self.list_comp,self.neighbor_mat=initial_merge(self.list_comp,self.neighbor_mat)
        self.color_sim=ColorSimilarity(self.neighbor_mat.shape)
        self.texture_sim=TextureSimilarity(self.neighbor_mat,img)
        self.size_sim=SizeSimilarity(self.neighbor_mat)
        self.fill_sim=FillSimilarity(self.neighbor_mat)
        self.color_sim.compute_similar(self.neighbor_mat,img,self.list_comp)
        self.texture_sim.compute_similar(self.neighbor_mat,self.list_comp)
        self.size_sim.compute_similar(self.neighbor_mat,self.list_comp,img.shape[0]*img.shape[1])
        self.fill_sim.compute_similar(self.neighbor_mat,self.list_comp,img.shape[0]*img.shape[1])
    def merge(self,coef,img_area):
        '''
        Merge these components
        Parameters:
        list_comp: list of components
        coef: coefficient of each similarity matrix
        '''
        # Calculate overall similarity matrix and find the max value
        self.sim=coef[0]*self.color_sim.sim_mat+coef[1]*self.texture_sim.sim_mat+coef[2]*self.size_sim.sim_mat+coef[3]*self.fill_sim.sim_mat
        pos=np.where(self.sim==np.max(self.sim))
        pos=(pos[0][0],pos[1][0])
        pos=(min(pos),max(pos))
        # Update similarity matrix in each class
        self.color_sim.update(pos,self.neighbor_mat,self.list_comp)
        self.texture_sim.update(pos,self.neighbor_mat,self.list_comp)
        self.size_sim.update(pos,self.neighbor_mat,self.list_comp,img_area)
        self.fill_sim.update(pos,self.neighbor_mat,self.list_comp,img_area)
        # Update and remove in list_comp, neighboring_mat
        new_comp=self.list_comp[pos[0]]+self.list_comp[pos[1]]
        self.list_comp.append(new_comp)
        self.list_comp.pop(pos[1])
        self.list_comp.pop(pos[0])
        self.neighbor_mat=np.insert(self.neighbor_mat,self.neighbor_mat.shape[0],np.zeros((self.neighbor_mat.shape[1],)).astype(bool),axis=0)
        self.neighbor_mat=np.insert(self.neighbor_mat,self.neighbor_mat.shape[1],np.zeros((self.neighbor_mat.shape[0],)).astype(bool),axis=1)
        self.neighbor_mat[-1,:]=np.logical_or(self.neighbor_mat[pos[0],:],self.neighbor_mat[pos[1],:])
        self.neighbor_mat[:,-1]=self.neighbor_mat[-1,:]
        self.neighbor_mat=np.delete(self.neighbor_mat,pos[1],axis=0)
        self.neighbor_mat=np.delete(self.neighbor_mat,pos[0],axis=0)
        self.neighbor_mat=np.delete(self.neighbor_mat,pos[1],axis=1)
        self.neighbor_mat=np.delete(self.neighbor_mat,pos[0],axis=1)
    def merge_regions(self,strategy,img_area,nregions=100):
        '''
        Merge regions ultil the merged regions=100
        Strategy is a tuple, represent the method used for calculating similarity matrix
        '''
        count=0
        while(count<nregions):
            print('Merge for {} regions ...'.format(count+1))
            self.merge(coef=strategy,img_area=img_area)
            count+=1
    
################

################

def retrieve_components(img,file_name='result.txt'):
    '''
    Retrieve components from image
    Parameters:
    img: image
    file_name: components file
    '''
    print('Retrieving components from {} ...'.format(file_name))
    img_comp=np.zeros((img.shape[0],img.shape[1])).astype(int)
    list_comp=[]
    with open(file_name,'r') as f:
        line=f.readline()
        while(line!=''):
            split_line=line.split(' ')
            comp_idx=int(split_line[0])
            current_comp=[]
            for pixel in split_line[1:-1]:
                pixel_pos=pixel.split(',')
                pixel_y=int(pixel_pos[0])
                pixel_x=int(pixel_pos[1])
                img_comp[pixel_y,pixel_x]=comp_idx
                current_comp.append((pixel_y,pixel_x))
            list_comp.append(current_comp)
            line=f.readline()
    return img_comp,list_comp
def initial_segment(list_comp,img_shape):
    '''
    Initialize the segmented image, based on list of components
    '''
    np.random.seed(10000)
    # color=np.array([[10,10,250],[10,250,10],[250,10,10],[120,120,120],[120,40,230],[40,120,230],[230,40,120],[80,150,130],[130,60,70]])
    segmented_img=np.zeros((img_shape[0],img_shape[1],3)).astype(np.uint8)
    for idx,component in enumerate(list_comp):
        color=np.random.randint(low=0,high=255,size=(3,))
        for pos in component:
            segmented_img[pos[0],pos[1]]=color
    return segmented_img
def find_neighbors(img_comp,max_numb):
    '''
    Find the neighboring components in the image
    img_comp: the 2D numpy array for storing the components in each pixel
    '''
    print('Finding components neighbors ...')
    mat=np.zeros((max_numb,max_numb)).astype(bool)
    for y in range(img_comp.shape[0]):
        for x in range(img_comp.shape[1]):
            sub_area=img_comp[max(0,y-1):min(img_comp.shape[0],y+2),max(0,x-1):min(img_comp.shape[1],x+2)]
            if(len(np.unique(sub_area))==1):continue
            else:
                for value in np.unique(sub_area):mat[img_comp[y,x],value]=True
    return mat
def initial_merge(list_comp,neighbor_mat):
    '''
    Start to merge small regions, regions which have 1-3 pixels
    '''
    print('Starting initial merge ...')
    idx=0
    while(idx<len(list_comp)):
        if(len(list_comp[idx])<=18):
            flag=True
            pos=np.where(neighbor_mat[idx,:]==True)[0]
            for j in pos:
                if(len(list_comp[j])>=40):
                    '''
                    We need to merge these components
                    '''
                    neighbor_mat[j,:]=np.logical_or(neighbor_mat[j,:],neighbor_mat[idx,:])
                    neighbor_mat[:,j]=neighbor_mat[j,:]
                    neighbor_mat=np.delete(neighbor_mat,idx,0)
                    neighbor_mat=np.delete(neighbor_mat,idx,1)
                    list_comp[j].extend(list_comp[idx])
                    list_comp.pop(idx)
                    flag=False
                    break
            if(flag):idx+=1
        else:idx+=1
    return list_comp,neighbor_mat