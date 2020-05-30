//source code

#include <iostream>
#include <opencv2/opencv.hpp>
#include "segmented.h"

/*
Gaussian blur function
*/
cv::Mat gaussian_blur(cv::Mat& img, float sigma){
    // GaussianBlur function
    // Parameters: img: input image
    // sigma: standard deviation, here we consider sigma equal in both x and y direction

    int kernel_size = 2*(int)(3*sigma)+1;
    cv::Mat dst;
    cv::GaussianBlur(img,dst,cv::Size(),sigma,sigma);
    return dst;
}

/*
Construct vertice from pixel coordinates
*/

Vertice construct_vertice(int y_coord, int x_coord){
    //construct vertice function
    //Parameters: y_coord: y coordinate of that vertice
    // x_coord: x coordinate of that vertice
    Vertice ver;
    ver.y_coord=y_coord;
    ver.x_coord=x_coord;
    return ver;
}

/*
Construct edge from neighboring vertice
*/

Edge construct_edge(Vertice first_ver, Vertice second_ver, float weight){
    //construct edge from Vertices
    //Parameters:
    // first_ver: the first vertice
    // second_ver: the second vertice
    // weight: weight of that edge
    Edge edge;
    edge.first_ver = first_ver;
    edge.second_ver = second_ver;
    edge.weight = weight;
    return edge;
}

/*
Class Graph
*/

Graph::Graph(){} //default constructor

// form graph

void Graph:: form_graph(cv::Mat& img){
    //initialize the array
    std::cout<<"Form Graph ...";
    len_graph=4*(img.rows-2)*(img.cols-2)+6+5*(img.rows-2)+5*(img.cols-2);
    arr=new Edge[len_graph];
    int pos_arr=0;
    Vertice ver1;
    Vertice ver2;
    for(int row=0;row<img.rows;row++){
        for(int col=0;col<img.cols;col++){
            if(col+1<img.cols){
                ver1=construct_vertice(row,col);
                ver2=construct_vertice(row,col+1);
                arr[pos_arr]=construct_edge(ver1,ver2,fabs(img.at<uint8_t>(row,col)-img.at<uint8_t>(row,col+1)));
                pos_arr++;
            }
            if(row+1<img.rows){
                ver1=construct_vertice(row,col);
                ver2=construct_vertice(row+1,col);
                arr[pos_arr]=construct_edge(ver1,ver2,fabs(img.at<uint8_t>(row,col)-img.at<uint8_t>(row+1,col)));
                pos_arr++;
            }
            if(row+1<img.rows && col+1<img.cols){
                ver1=construct_vertice(row,col);
                ver2=construct_vertice(row+1,col+1);
                arr[pos_arr]=construct_edge(ver1,ver2,fabs(img.at<uint8_t>(row,col)-img.at<uint8_t>(row+1,col+1)));
                pos_arr++;
            }
            if(row+1<img.rows && col-1>=0){
                ver1=construct_vertice(row,col);
                ver2=construct_vertice(row+1,col-1);
                arr[pos_arr]=construct_edge(ver1,ver2,fabs(img.at<uint8_t>(row,col)-img.at<uint8_t>(row+1,col-1)));
                pos_arr++;
            }
        }
    }
    std::cout<<"Done"<<std::endl;
}

/*
Heap class
*/

//constructor
Heap::Heap(Edge* edge_arr, long length_heap){
    arr = edge_arr;
    length = length_heap;
}

//switch_arr method
void Heap::switch_arr(long i,long j){
    //switch 2 position in arr
    /*
    Parameters:
    i,j: two position we want to switch
    */
    Edge temp=arr[i];
    arr[i]=arr[j];
    arr[j]=temp;
}

// enHeap method

void Heap::enHeap(long i){
    /*
    enHeap at position i
    Parameters:
    i: the position we want to enHeap
    */
   long parent=i;
   while(2*parent+1<length){
       //while we're not out of Heap
       if(2*parent+2==length){
           //if there is only one child node of the current node
           long leftChild=2*parent+1;
           if(arr[leftChild].weight<arr[parent].weight)switch_arr(parent,leftChild);
           break;
       }
       else{
           long leftChild=2*parent+1;
           long rightChild=2*parent+2;
           long temp;
           if(arr[leftChild].weight<=arr[rightChild].weight)temp=leftChild;
           else temp=rightChild;
           if(arr[parent].weight>arr[temp].weight){
               switch_arr(parent,temp);
               parent=temp;
            }
            else break;
           }
       }
   }

//enFullHeap

void Heap::enFullHeap(){
    std::cout<<"Starting to enHeap ...";
    for(long i=(length-1)/2;i>=0;i--)enHeap(i);
    std::cout<<"Done"<<std::endl;
}

//pop Heap method
Edge Heap::pop(){
    Edge edge = arr[0]; //the return value of method
    switch_arr(0,length-1); //switch the first and the last of Heap
    length--; //change the value of length
    enHeap(0); //reenHeap
    return edge;
}

/*
GraphBasedImpl class: Implementation the Algorithm
*/

//constructor

GraphBasedImpl::GraphBasedImpl(cv::Mat& img){
    //initialize the forest array of the image
    forest = new Vertice*[img.rows];
    for(int row=0;row<img.rows;row++){
        forest[row] = new Vertice[img.cols];
    }
    //intialize the comp matrix
    comp = new int*[img.rows];
    for(int row=0;row<img.rows;row++){
        comp[row] = new int[img.cols];
    }
}

//predicate function
bool GraphBasedImpl::predicate(double weight, Vertice& C1, Vertice& C2, double k){
    // Determine if there is an evidence about a boundary between component C1 and component C2
    // Return true if there is an evidence, else return false
    // Parameters:
    // weight: weight of the considering edge
    // C1: the value of the forest array at the position of component C1, first value is a negative number
    // represents the size of that component, the second value is a positive number, represent the maximum
    // weight of the minimum spanning tree of that component
    // C2: same as C1, but for the component 2
    // k: hyperparameters for determining the threshold function
    double mint_C1 = C1.x_coord + fabs(k/C1.y_coord);
    double mint_C2 = C2.x_coord + fabs(k/C2.y_coord);
    double mint = std::min(mint_C1, mint_C2);
    if(weight > mint) return true;
    else return false;
}

//getRoot function
Vertice GraphBasedImpl::getRoot(Vertice& ver){
    //get the root vertice of given ver
    Vertice* temp=&ver; //create a pointer to ver
    while (forest[temp->y_coord][temp->x_coord].y_coord >= 0){
        temp=&forest[temp->y_coord][temp->x_coord];
    }
    Vertice new_ver = construct_vertice(temp->y_coord,temp->x_coord);
    return new_ver;
}

//merge_component

void GraphBasedImpl::merge_comp(Vertice& root1, Vertice& root2, int weight){
    //Merge the two components
    //Parameters:
    // root1: root node represents the component 1
    // root2: root node represents the component 2
    // weight: weight of the considering edge
    long size1 = -forest[root1.y_coord][root1.x_coord].y_coord; //size of component 1
    long size2 = -forest[root2.y_coord][root2.x_coord].y_coord; //size of component 2
    if(size1>=size2){
        //if size1 is greater, we merge the component 2 to component 1
        forest[root2.y_coord][root2.x_coord]=root1;
        Vertice new_ver = construct_vertice(-(size1+size2),weight);
        forest[root1.y_coord][root1.x_coord]=new_ver;
    }
    else{
        //if size2 is greater, then we merge the component 1 to component 2
        forest[root1.y_coord][root1.x_coord]=root2;
        Vertice new_ver = construct_vertice(-(size1+size2),weight);
        forest[root2.y_coord][root2.x_coord] = new_ver;
    }
}

void GraphBasedImpl::form_components(cv::Mat& img_channel, double k){
    //form component for an image channel and given k parameters
    //firstly, we must initialize the forest
    Vertice ver = construct_vertice(-1,0);
    for(int row=0; row<img_channel.rows; row++){
        for(int col=0; col < img_channel.cols; col++){
            forest[row][col]=ver;
        }
    }
    //construct graph and heap
    Graph graph;
    graph.form_graph(img_channel);
    Heap heap(graph.arr, graph.len_graph);
    heap.enFullHeap();
    //starting to form components
    std::cout<<"Form components ...";
    while(heap.length>0){
        Edge edge = heap.pop(); //pop the first edge in Heap
        Vertice root1 = getRoot(edge.first_ver); //root node of component 1
        Vertice root2 = getRoot(edge.second_ver); //root node of component 2
        if(root1.y_coord!=root2.y_coord || root1.x_coord!=root2.x_coord){
            //merge these two components
            if(!predicate(edge.weight,forest[root1.y_coord][root1.x_coord],forest[root2.y_coord][root2.x_coord],k))
            merge_comp(root1,root2,edge.weight);
        }
    }
    std::cout<<"Done"<<std::endl;
}

//create segment function

void GraphBasedImpl::create_segment(cv::Mat& img, float sigma, double k){
    //create the segmentation for the image
    //Parameters: img: the image
    cv::Mat new_img = gaussian_blur(img,sigma);
    cv::Mat *channel = new cv::Mat[3]; //create an array for 3 channel for mat
    split(new_img,channel); //split the image into 3 seperate channels
    //initialize the segmented image
    segmented_img = cv::Mat(img.rows,img.cols,CV_8UC3,cv::Scalar(0,0,0));
    std::cout<<segmented_img.channels()<<std::endl;
    uint8_t color[4] = {10,75,150,240};
    //for each channel, we form the components and retrieve the components matrix
    form_components(channel[0],k);//form components for channel 1
    retrieve_components(img);
    for(int row = 0;row<img.rows;row++){
        for(int col = 0;col<img.cols;col++){
            cv::Vec3b &my_color = segmented_img.at<cv::Vec3b>(row,col);
            my_color[0] = color[comp[row][col]%4];
        }
    }
    std::cout<<"Finish"<<std::endl;
    form_components(channel[1],k); //form components for channel 2
    retrieve_components(img);
    for(int row = 0;row<img.rows;row++){
        for(int col=0; col<img.cols;col++){
            cv::Vec3b &my_color = segmented_img.at<cv::Vec3b>(row,col);
            my_color[1] = color[comp[row][col]%4];
        }
    }
    std::cout<<"Finish"<<std::endl;
    form_components(channel[2],k); //form components for channel 3
    retrieve_components(img);
    for(int row = 0;row<img.rows;row++){
        for(int col=0; col<img.cols;col++){
            cv::Vec3b &my_color = segmented_img.at<cv::Vec3b>(row,col);
            my_color[2] = color[comp[row][col]%4];
        }
    }
    std::cout<<"Finish"<<std::endl;
}

//retrieve components function

void GraphBasedImpl::retrieve_components(cv::Mat& img){
    // the forest is convinient for us to Traceback, but not convinient for us to retrieve components

    // we need to assign to each component a number
    //Parameters: img: the image Mat
    //Return: an array that label 0,1,.. for each pixels for its component
    std::cout<<"Retrieving components from forest ...";

    //after initialize the component matrix, we assign the label 0,1 to the root node pixels
    //this is achived by considering the y_coord smaller than 0 in the forest array
    int index=0;
    for(int row=0;row<img.rows;row++){
        for(int col=0;col<img.cols;col++){
            if(forest[row][col].y_coord<0){
                comp[row][col]=index;
                index++;
            }
        }
    }

    //after this step, we have assigned the label for root node
    //next, we will assign the label for other node
    for(int row=0;row<img.rows;row++){
        for(int col=0;col<img.cols;col++){
            Vertice ver = construct_vertice(row,col);
            Vertice root = getRoot(ver);
            comp[row][col] = comp[root.y_coord][root.x_coord];
        }
    }
    std::cout<<"Done"<<std::endl;
}

//get segmented image function

cv::Mat GraphBasedImpl::get_segmented_image() const&{
    return segmented_img;
}

//write image to file
void GraphBasedImpl::write_image(cv::String img_name){
    cv::imwrite(img_name,segmented_img);
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

