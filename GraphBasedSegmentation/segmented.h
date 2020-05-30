
//header file

#include <opencv2/opencv.hpp>

//Gaussian Blur function

cv::Mat gaussian_blur(cv::Mat& ,float);

//define the struct Vertice

struct Vertice{
    int y_coord;
    int x_coord;
};

//construct vertice from point coordinates

Vertice construct_vertice(int,int);

//define struct edge

struct Edge{
    Vertice first_ver;
    Vertice second_ver;
    float weight;
};

//construct edge from 2 vertice

Edge construct_edge(Vertice,Vertice,float);

// define Graph class

class Graph{
    public:
       Edge* arr; //array of edge
       Graph(); //constructor
       long len_graph; //len of edge array
       void form_graph(cv::Mat&); //form graph method
};

// define the Heap class

class Heap{
    private:
       Edge* arr; //Heap array
       void switch_arr(long,long); //switch array element of Heap
       void enHeap(long); //enHeap
    public:
       long length; //length of Heap
       Heap(Edge*, long); //constructor
       void enFullHeap();
       Edge pop(); //pop the first element from heap
};

// GraphBasedImpl class

class GraphBasedImpl{
    private:
        Vertice** forest; //forest array
        int** comp; //component array
        cv::Mat segmented_img; //segmented image result
        bool predicate(double, Vertice&, Vertice&, double); //the predicate function
        Vertice getRoot(Vertice& ver); //get root of component
        void merge_comp(Vertice&, Vertice&, int); //merge 2 components
        void form_components(cv::Mat&, double); //form component
        void retrieve_components(cv::Mat& img); //retrieve comp

    public:
        GraphBasedImpl(cv::Mat& img); //constructor
        cv::Mat get_segmented_image() const&; //get the segmented image
        void create_segment(cv::Mat&, float, double); //create the segmented image
        void write_image(cv::String img_name); //write the segmented image to file
};

std::string type2str(int); //return the type of image mat
