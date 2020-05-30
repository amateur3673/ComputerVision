#include <iostream>
#include "segmented.h"
using namespace cv;
using namespace std;

int main(int argc, char *argv[]){
    string image_link;
    cout<<"Enter image link: ";
    cin>>image_link;
    Mat img = imread(image_link,CV_LOAD_IMAGE_COLOR);
    if(!img.data){
        cout<<"Error when trying to open the image"<<endl;
        return -1;
    }
    double k;
    cout<<"Enter the value of k:";
    cin>>k;
    GraphBasedImpl impl(img);
    impl.create_segment(img,0.8,k);
    Mat segmented_img = impl.get_segmented_image();
    namedWindow("Babe",WINDOW_AUTOSIZE);
    imshow("Babe",segmented_img);
    waitKey(0);
}