import numpy as np
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
            return_img=np.zeros((3,)).astype(np.float64)
            for c in range(3):
                return_img[c]=round(img[y1,x1,c]*(x2-pos[1])+img[y2,x2,c]*(pos[1]-x1))
            return return_img
        elif(abs(x2-x1)<1e-7):
            return_img=np.zeros((3,)).astype(np.float64)
            for c in range(3):
                return_img[c]=round(img[y1,x1,c]*(y2-pos[0])+img[y2,x2,c]*(pos[0]-y1))
            return return_img
        else:
            return_img=np.zeros((3,)).astype(np.float64)
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
        new_img=np.zeros_like(img).astype(np.float64)
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