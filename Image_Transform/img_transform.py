import numpy as np
import cv2
def _nearest_interpolation(img,pos):
    if(len(img.shape)==3):return img[int(pos[0]),int(pos[1]),:]
    else: return img[int(pos[0]),int(pos[1])]

def _bilinear_interpolation(img,pos):
    '''
    Bilinear interpolation
    '''
    y1=int(np.floor(pos[0]))
    y2=int(np.ceil(pos[0]))
    x1=int(np.floor(pos[1]))
    x2=int(np.ceil(pos[1]))
    if(y2>=img.shape[0]):y2=img.shape[0]-1
    if(x2>=img.shape[1]):x2=img.shape[1]-1
    if(len(img.shape)==3):
        if(abs(y2-y1)<1e-7 and abs(x2-x1)<1e-7):
            return img[y1,x1,:]
        elif(abs(y2-y1)<1e-7):
            return_img=np.zeros((3,)).astype(np.uint8)
            for c in range(3):
                return_img[c]=round(img[y1,x1,c]*(x2-pos[1])+img[y2,x2,c]*(pos[1]-x1))
            return return_img
        elif(abs(x2-x1)<1e-7):
            return_img=np.zeros((3,)).astype(np.uint8)
            for c in range(3):
                return_img[c]=round(img[y1,x1,c]*(y2-pos[0])+img[y2,x2,c]*(pos[0]-y1))
            return return_img
        else:
            return_img=np.zeros((3,)).astype(np.uint8)
            x_mat=np.array([[x2-pos[1]],[pos[1]-x1]])
            y_mat=np.array([[y2-pos[0],pos[0]-y1]])
            for c in range(3):
                value_mat=np.array([[img[int(y1),int(x1),c],img[int(y1),int(x2),c]],[img[int(y2),int(x1),c],img[int(y2),int(x2),c]]])
                return_img[c]=round((y_mat.dot(np.dot(value_mat,x_mat)))[0,0])
            return return_img
    else:
        if(abs(y2-y1)<1e-7 and abs(x2-x1)<1e-7):return img[y1,x1]
        elif(abs(y2-y1)<1e-7):
            return round(img[y1,x1]*(x2-pos[1])+img[y2,x2]*(pos[1]-x1))
        elif(abs(x2-x1)<1e-7):
            return round(img[y1,x1]*(y2-pos[0])+img[y2,x2]*(pos[0]-y1))
        else:
            x_mat=np.array([[x2-pos[1]],[pos[1]-x1]])
            y_mat=np.array([[y2-pos[0],pos[0]-y1]])
            value_mat=np.array([[img[int(y1),int(x1),c],img[int(y1),int(x2),c]],[img[int(y2),int(x1),c],img[int(y2),int(x2),c]]])
            return round((y_mat.dot(np.dot(value_mat,x_mat)))[0,0])

def image_scaling(img,new_shape,interpolation='NEAREST_NEIGHBOR'):
    '''
    Perform image scaling
    img: input image
    new_shape: new shape for this image
    method: method for image scaling
    '''
    scale_factor=(new_shape[0]/img.shape[0],new_shape[1]/img.shape[1])
    old_pos_x=np.arange(new_shape[1])/scale_factor[1]
    old_pos_y=np.arange(new_shape[0])/scale_factor[0]
    if(interpolation=='NEAREST_NEIGHBOR'):
      if(len(img.shape)==3):
        new_img=np.zeros((new_shape[0],new_shape[1],3)).astype(np.uint8)
        for y in range(new_shape[0]):
            for x in range(new_shape[1]):
                new_img[y,x,:]=img[int(old_pos_y[y]),int(old_pos_x[x]),:]
      else:
        new_img=np.zeros((new_shape[0],new_shape[1])).astype(np.uint8)
        for y in range(new_shape[0]):
            for x in range(new_shape[1]):
                new_img[y,x]=img[int(old_pos_y[y]),int(old_pos_x[x])]
    elif(interpolation=='BILINEAR'):
        if(len(img.shape)==3):
            new_img=np.zeros((new_shape[0],new_shape[1],3)).astype(np.uint8)
            for y in range(1,new_shape[0]):
                for x in range(new_shape[1]):
                    new_img[y,x,:]=_bilinear_interpolation(img,(y/scale_factor[0],x/scale_factor[1]))
        else:
            new_img=np.zeros((new_shape[0],new_shape[1])).astype(np.uint8)
            for y in range(new_shape[0]):
                for x in range(new_shape[1]):
                    new_img[y,x]=_bilinear_interpolation(img,(y/scale_factor[0],x/scale_factor[1]))
    return new_img
class rotation_transform:
    '''
    Perform image rotation
    '''
    def __init__(self,img,center,angle,interpolation='NEAREST'):
        '''
        Initialize object
        Params:
        img: image
        center: center of the rotation
        angle: rotating an angle clockwise
        interpolation: interpolation method, includes 'NEAREAST' for nearest neighbor
        and 'BILINEAR' for bilinear interpolation
        '''
        self.center=center
        self.img=img
        self.rot_img=self._rotation(img,center,angle,interpolation)
    def _rotation(self,img,center,angle,interpolation):
        new_img=np.zeros_like(img).astype(np.uint8)
        self.angle=angle*np.pi/180
        self.rotation_mat=np.array([[np.cos(self.angle),-np.sin(self.angle),0],[np.sin(self.angle),np.cos(self.angle),0],[0,0,1]])
        self.inv_rot=np.linalg.inv(self.rotation_mat)
        if(interpolation=='NEAREST'):
            if(len(img.shape)==3):
                for y in range(img.shape[0]):
                    for x in range(img.shape[1]):
                        coord=np.array([[x-center[1]],[y-center[0]],[1]])
                        old_coord=self.inv_rot.dot(coord)
                        pos=[old_coord[1,0]+center[0],old_coord[0,0]+center[1]]
                        if(pos[0]>=0 and pos[0]<img.shape[0] and pos[1]>=0 and pos[1]<img.shape[1]):
                            new_img[y,x,:]=_nearest_interpolation(img,pos)
            else:
                for y in range(img.shape[0]):
                    for x in range(img.shape[1]):
                        coord=np.array([[x-center[1]],[y-center[0]],[1]])
                        old_coord=self.inv_rot.dot(coord)
                        pos=[old_coord[1,0]+center[0],old_coord[0,0]+center[1]]
                        if(pos[0]>=0 and pos[0]<img.shape[0] and pos[1]>=0 and pos[1]<img.shape[1]):
                            new_img[y,x]=_nearest_interpolation(img,pos)
        elif(interpolation=='BILINEAR'):
            if(len(img.shape)==3):
                for y in range(img.shape[0]):
                    for x in range(img.shape[1]):
                        coord=np.array([[x-center[1]],[y-center[0]],[1]])
                        old_coord=self.inv_rot.dot(coord)
                        pos=[old_coord[1,0]+center[0],old_coord[0,0]+center[1]]
                        if(pos[0]>=0 and pos[0]<img.shape[0] and pos[1]>=0 and pos[1]<img.shape[1]):
                            new_img[y,x,:]=_bilinear_interpolation(img,pos)
            else:
                for y in range(img.shape[0]):
                    for x in range(img.shape[1]):
                        coord=np.array([[x-center[1]],[y-center[0]],[1]])
                        old_coord=self.inv_rot.dot(coord)
                        pos=[old_coord[1,0]+center[0],old_coord[0,0]+center[1]]
                        if(pos[0]>=0 and pos[0]<img.shape[0] and pos[1]>=0 and pos[1]<img.shape[1]):
                            new_img[y,x]=_nearest_interpolation(img,pos)
        return new_img
    def _tobox(self,interpolation='NEAREST'):
        '''
        Find the bounding box to cover all the rotated image
        '''
        top_left=np.array([[-self.center[1]],[-self.center[0]],[1]])
        top_right=np.array([[self.rot_img.shape[1]-self.center[1]],[-self.center[0]],[1]])
        bottom_left=np.array([[-self.center[1]],[self.rot_img.shape[0]-self.center[0]],[1]])
        bottom_right=np.array([[self.rot_img.shape[1]-self.center[1]],[self.rot_img.shape[0]-self.center[0]],[1]])
        top_left_trans=self.rotation_mat.dot(top_left)
        top_right_trans=self.rotation_mat.dot(top_right)
        bottom_left_trans=self.rotation_mat.dot(bottom_left)
        bottom_right_trans=self.rotation_mat.dot(bottom_right)
        trans_coord=np.array([[top_left_trans[0,0],top_right_trans[0,0],bottom_left_trans[0,0],bottom_right_trans[0,0]],[top_left_trans[1,0],top_right_trans[1,0],bottom_left_trans[1,0],bottom_right_trans[1,0]]])
        top=np.min(trans_coord[1,:])
        bottom=np.max(trans_coord[1,:])
        left=np.min(trans_coord[0,:])
        right=np.max(trans_coord[0,:])
        height=int(bottom)-int(top)
        width=int(right)-int(left)
        center=(-int(top),-int(left))
        if(interpolation=='NEAREST'):
            if(len(self.img.shape)==3):
                new_img=np.zeros((height,width,3)).astype(np.uint8)
                for y in range(height):
                    for x in range(width):
                       coord=np.array([[x-center[1]],[y-center[0]],[1]])
                       old_coord=self.inv_rot.dot(coord)
                       pos=[old_coord[1,0]+self.center[0],old_coord[0,0]+self.center[1]]
                       if(pos[0]>=0 and pos[0]<self.img.shape[0] and pos[1]>=0 and pos[1]<self.img.shape[1]):
                          new_img[y,x,:]=_nearest_interpolation(self.img,pos)
            else:
                new_img=np.zeros((height,width)).astype(np.uint8)
                for y in range(height):
                    for x in range(width):
                        coord=np.array([[x-center[1]],[y-center[0]],[1]])
                        old_coord=self.inv_rot(coord)
                        pos=[old_coord[1,0]+self.center[0],old_coord[0,0]+self.center[1]]
                        if(pos[0]>=0 and pos[0]<self.img.shape[0] and pos[1]>=0 and pos[1]<self.img.shape[1]):
                            new_img[y,x]=_nearest_interpolation(self.img,pos)
        if(interpolation=='BILINEAR'):
            if(len(self.img.shape)==3):
                new_img=np.zeros((height,width,3)).astype(np.uint8)
                for y in range(height):
                    for x in range(width):
                       coord=np.array([[x-center[1]],[y-center[0]],[1]])
                       old_coord=self.inv_rot.dot(coord)
                       pos=[old_coord[1,0]+self.center[0],old_coord[0,0]+self.center[1]]
                       if(pos[0]>=0 and pos[0]<self.img.shape[0] and pos[1]>=0 and pos[1]<self.img.shape[1]):
                          new_img[y,x,:]=_bilinear_interpolation(self.img,pos)
            else:
                new_img=np.zeros((height,width)).astype(np.uint8)
                for y in range(height):
                    for x in range(width):
                        coord=np.array([[x-center[1]],[y-center[0]],[1]])
                        old_coord=self.inv_rot(coord)
                        pos=[old_coord[1,0]+self.center[0],old_coord[0,0]+self.center[1]]
                        if(pos[0]>=0 and pos[0]<self.img.shape[0] and pos[1]>=0 and pos[1]<self.img.shape[1]):
                            new_img[y,x]=_bilinear_interpolation(self.img,pos)
        return new_img
class scale_transform:
    def __init__(self,img,center,scale_factor,interpolation='NEAREST'):
        self.transform_img=self._transform(img,center,scale_factor,interpolation)
    def _transform(self,img,center,scale_factor,interpolation):
        new_img=np.zeros_like(img).astype(np.uint8)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                coord=(y-center[0],x-center[1])
                old_coord=(coord[0]/scale_factor[0]+center[0],coord[1]/scale_factor[1]+center[1])
                if(interpolation=='NEAREST'):new_img[y,x]=_nearest_interpolation(img,old_coord)
                elif(interpolation=='BILINEAR'):new_img[y,x]=_bilinear_interpolation(img,old_coord)
        return new_img
class shear_transform:
    def __init__(self,img,center,shear_factor,interpolation='NEAREST'):
        self.transform_img=self._transform(img,center,shear_factor,interpolation)
    def _transform(self,img,center,shear_factor,interpolation):
        shear_mat=np.array([[1,shear_factor[0],0],[shear_factor[1],1,0],[0,0,1]])
        inv_shear_mat=np.linalg.inv(shear_mat)
        new_img=np.zeros_like(img).astype(np.uint8)
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                coord=np.array([[x-center[1]],[y-center[0]],[1]])
                old_coord=inv_shear_mat.dot(coord)
                pos=(old_coord[1,0]+center[0],old_coord[0,0]+center[1])
                if(pos[0]>=0 and pos[0]<img.shape[0] and pos[1]>=0 and pos[1]<img.shape[1]):
                    if(interpolation=='NEAREST'):new_img[y,x]=_nearest_interpolation(img,pos)
                    elif(interpolation=='BILINEAR'):new_img[y,x]=_bilinear_interpolation(img,pos)
        return new_img
if __name__=='__main__':
    img=cv2.imread('Images/Figure1.jpg')
    #img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    rot=rotation_transform(img,(img.shape[0]//2,img.shape[1]//2),45,interpolation='BILINEAR')
    img1=rot._tobox()
    rot=rotation_transform(img1,(img1.shape[0]//2,img1.shape[1]//2),-45,interpolation='BILINEAR')
    new_img=cv2.Scharr(rot.rot_img,cv2.CV_32F,0,1)
    new_size=rot.rot_img.shape
    old_size=img.shape
    offset_0=(new_size[0]-old_size[0])//2
    offset_1=(new_size[1]-old_size[1])//2
    crop=new_img[offset_0:rot.rot_img.shape[0]-offset_0,offset_1:rot.rot_img.shape[1]-offset_1,:]
    cv2.imshow('image',crop)
    cv2.waitKey(0)
    '''cv2.imshow('image',rot._tobox())
    cv2.waitKey(0)'''