import numpy as np
import cv2
import matplotlib.pyplot as plt
def convolve(img,kernel):
    '''
    Convolutional Operator between the origin_img and kernel
    Parameters:
    img: image, a 2D numpy array.
    kernel: the filter.
    return:
    convolved_img: the numpy array after convolving.
    '''
    #We want to keep the shape as close as possible to the image shape, so we use zero_padding
    pad_h=(kernel.shape[0]-1)//2 #pad at h dimension
    pad_w=(kernel.shape[1]-1)//2 #pad at w dimension
    #Calculate the new shape of the image
    convolved_h=img.shape[0]-kernel.shape[0]+2*pad_h+1
    convolved_w=img.shape[1]-kernel.shape[1]+2*pad_w+1
    #Use zero padding
    img=np.pad(img,pad_width=((pad_h,pad_h),(pad_w,pad_w)),mode='constant',constant_values=(0,0))
    # Initialize the new convolved_img
    convolved_img=np.zeros((convolved_h,convolved_w)).astype(int)
    for h in range(convolved_h):
        for w in range(convolved_w):
            slice=img[h:h+kernel.shape[0],w:w+kernel.shape[1]]
            convolved_img[h,w]=int(np.sum(slice*kernel))
            if(convolved_img[h,w]<0):convolved_img[h,w]%=255
            elif(convolved_img[h,w]>255):convolved_img[h,w]%=255
    return convolved_img
def create_gauss_filter(sigma=0.8,kernel_size=5):
    '''
    Create a filter for Gaussian filter.
    Parameters:
    sigma: standard deviation for the Gaussian Filter.
    kernel_size: kernel size for the filter
    return:
    a filter with shape=(kernel_size,kernel_size)
    '''
    #Initialize the kernel
    kernel=np.zeros((kernel_size,kernel_size))
    n_H=kernel_size//2 #height from -n_H to n_H
    n_W=kernel_size//2 #width from -n_W to n_W
    for h in range(-n_H,n_H+1):
        for w in range(-n_W,n_W+1):
            first_partition=1/(2*np.pi*(sigma**2))
            second_partition=np.exp(-(h**2+w**2)/(2*(sigma**2)))
            kernel[h+n_H,w+n_W]=first_partition*second_partition
    return kernel
def gaussian_filter(img,sigma,kernel_size=5):
    '''
    Apply the Gaussian Filter to the Image
    img: image, can be 2D array for grayscale image or 3D array for RGB Image
    sigma: standard deviation for Gaussian Filter
    kernel_size: size of kernel
    return:
    filter image
    '''
    kernel=create_gauss_filter(sigma,kernel_size) #Create a mask
    if(len(img.shape)==2):
        #This case is grayscale image
        img_filter=convolve(img,kernel)
    else:
        #RGB Image
        img_filter1=convolve(img[:,:,0],kernel)
        img_filter2=convolve(img[:,:,1],kernel)
        img_filter3=convolve(img[:,:,2],kernel)
        img_filter=np.zeros((img_filter1.shape[0],img_filter1.shape[1],3)).astype(int)
        img_filter[:,:,0]=img_filter1
        img_filter[:,:,1]=img_filter2
        img_filter[:,:,2]=img_filter3
    return img_filter
class Edge:
    '''
    Edge class in Graph.
    '''
    def __init__(self,first_ver,second_ver,weight):
        '''
        Initialize an edge.
        Parameters:
        first_ver: a tuple represents the first vertice of the Edge
        second_ver: a tuple represents the second vertice of the Edge
        weight: weight on that Edge
        '''
        self.first=first_ver
        self.second=second_ver
        self.weight=weight
def construct_graph(img):
    '''
    Constructing Graph from img
    Parameters:
    img: 2D array for constructing Graph
    return:
    list of edges represent the constructed Graph
    '''
    graph=[] #Edge list
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(j+1<img.shape[1]):
                edge=Edge((i,j),(i,j+1),np.abs(img[i,j+1]-img[i,j]))
                graph.append(edge)
            if(i+1<img.shape[0]):
                edge=Edge((i,j),(i+1,j),np.abs(img[i+1,j]-img[i,j]))
                graph.append(edge)
            if(i+1<img.shape[0] and j+1<img.shape[1]):
                edge=Edge((i,j),(i+1,j+1),np.abs(img[i+1,j+1]-img[i,j]))
                graph.append(edge)
            if(i+1<img.shape[0] and j-1>=0):
                edge=Edge((i,j),(i+1,j-1),np.abs(img[i+1,j-1]-img[i,j]))
                graph.append(edge)
    return graph
class Sort:
    '''
    Quicksort class for sorting the edge in non-decreasing weight order.
    '''
    def __init__(self,graph):
        '''
        Initialize, graph is edge list Graph
        '''
        self.graph=graph
    def quick(self,S,F):
        # Quick sort the array from position S to F
        if(S<F):
            i=S;j=F
            pivot=self.graph[(S+F)//2].weight
            while(i<=j):
                while(self.graph[i].weight<pivot):i+=1
                while(self.graph[j].weight>pivot):j-=1
                if(i<=j):
                    if(i<j):
                        temp=self.graph[i]
                        self.graph[i]=self.graph[j]
                        self.graph[j]=temp
                    i+=1
                    j-=1
            self.quick(S,j)
            self.quick(i,F)
    def sort(self):
        self.quick(0,len(self.graph)-1)
        return self.graph
# Construct spanning tree
# Forest is a matrix, where (i,j) represent the vertice (i,j).
# If first value is negative, then it is a root node of tree in 1 component, the the second value of the tuple 
# represent the maximum edge of the minimum spanning tree formed by that componet.
def predicate(weight,C1,C2,k=600):
    '''
    pairwise comparison predicate.
    Parameters:
    weight: the weight of the considering vertice
    C1: tuple,value of the root node of the first vertice, the first value in the tuple is a negative
    number, represent the size of that component, the second value of that tuple is non-negative number
    represents the maximum weight of the minimum spanning tree in that component
    C2: same as C1
    k: the hyperparameters in the algorithm
    '''
    mint_C1=C1[1]+abs(k/C1[0])
    mint_C2=C2[1]+abs(k/C2[0])
    mint=min(mint_C1,mint_C2)
    if(weight>mint):return True
    else: return False
def getRoot(vertice,forest):
    '''
    get the root of that vertice in the minimum spanning tree built in that vertice component
    Parameters:
    vertice: tuple, represent the vertice
    forest: array of forest (defined above)
    return the root node
    '''
    while(forest[vertice[0]][vertice[1]][0]>=0):
        vertice=forest[vertice[0]][vertice[1]]
    return vertice
def merge(root_1,root_2,weight,forest):
    '''
    Merge 2 components comprises of root_1 and root_2
    Parameters:
    root_1:tuple, the first component
    root_2:tuple, the second component
    forest: forest array
    '''
    size_1=-forest[root_1[0]][root_1[1]][0]
    size_2=-forest[root_2[0]][root_2[1]][0]
    mst_1=forest[root_1[0]][root_1[1]][1] #the maximum weight of the minimum spanning tree in component 1
    mst_2=forest[root_2[0]][root_2[1]][1] #the maximum weight of the minimum spanning tree in component 2
    if(size_1>=size_2):
        #Merge component 2 to component 1
        forest[root_2[0]][root_2[1]]=root_1 #the root_2 point to root_1
        forest[root_1[0]][root_1[1]]=(-size_1-size_2,weight) #add the size and update the weight
    else:
        #Merge component 1 to component 2
        forest[root_1[0]][root_1[1]]=root_2 #root 1 points to root_2
        forest[root_2[0]][root_2[1]]=(-size_1-size_2,weight) #update the size and weight
def form_component(img,k=600):
    '''
    Form the component form img
    Parameters:
    img:image
    k: for control tau function
    return:
    forest
    '''
    graph=construct_graph(img)
    sort_graph=Sort(graph)
    graph=sort_graph.sort()
    forest=[[(-1,0) for i in range(img.shape[1])]for j in range(img.shape[0])]
    for edge in graph:
        vertice_1=edge.first
        vertice_2=edge.second
        weight=edge.weight
        root_1=getRoot(vertice_1,forest)
        root_2=getRoot(vertice_2,forest)
        if(root_1[0]!=root_2[0] or root_1[1]!=root_2[1]):
            if(not predicate(weight,C1=forest[root_1[0]][root_1[1]],C2=forest[root_2[0]][root_2[1]],k=k)):
                merge(root_1,root_2,weight,forest)
    return forest
def retrieve_component(forest):
    '''
    Retrive the component from forest
    Parameters:
    forest
    return: dictionary of component, keys are the root node, value are list of node in that component
    '''
    components={} #Initialize the list of the component
    for i in range(len(forest)):
        for j in range(len(forest[0])):
            root=getRoot((i,j),forest)
            if(not (root in components.keys())):
                components[root]=[]
                components[root].append((i,j))
            else: components[root].append((i,j))
    return components
def assign_segment(img,k=500):
    '''
    assign the segmentation of the image
    Parameters:
    img: the image
    return:
    assign color image (segmented image)
    '''
    color=(10,75,150,240)
    if(len(img.shape)==3):
        #RGB image
        segmented_image=np.zeros((img.shape[0],img.shape[1],3)).astype(np.uint8) #Initialize the segmented image
        for c in range(3):
            forest=form_component(img[:,:,c],k=k)
            components=retrieve_component(forest)
            j=0
            for _,nodes in components.items():
                for node in nodes:
                    segmented_image[node[0],node[1],c]=color[j%4]
                j+=1
    return segmented_image
def main(image_path='Images/Figure3.jpeg'):
    img=cv2.imread(image_path)
    img=img[:,:,::-1]
    img=gaussian_filter(img,sigma=0.8,kernel_size=5)
    segmented_image=assign_segment(img,k=20000)
    return segmented_image
if __name__=='__main__':
    segmented_image=main()
    plt.imshow(segmented_image)
    plt.show()
