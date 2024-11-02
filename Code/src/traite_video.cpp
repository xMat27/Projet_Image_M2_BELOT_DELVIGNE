// test_couleur.cpp : Seuille une image en niveau de gris

#include <stdio.h>
#include "image_ppm.h"
#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
  char cNomVideoLue[250], cNomVideoTraite[250];
  int taille_imagette, nbImagette, nbFrameIntacte;
  
  if (argc != 3) 
	{
		printf("Usage: VideoInitiale.mp4 VideoTraitée.mp4 \n"); 
		return 1;
	}

	sscanf (argv[1],"%s",cNomVideoLue) ;
  sscanf (argv[2],"%s",cNomVideoTraite) ;

  VideoCapture cap(cNomVideoLue);
  if (!cap.isOpened()) {
    std::cerr << "Erreur lors de l'ouverture de la vidéo" << std::endl;
    return -1;
  }else{
    std::cout << "La vidéo a bien été ouverte\n";
  }

  int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(CAP_PROP_FPS);
  int tailleFrame = frame_width * frame_height;

  VideoWriter videoTraite(cNomVideoTraite, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height), true);
  int nbFrame = 0;


  

  while (true) {
      Mat frame;
      if (!cap.read(frame)){
          break; // Toute les frames de la vidéo initiale ont été lu
      }

      OCTET *ImgIn, *ImgOut;
      allocation_tableau(ImgIn, OCTET, tailleFrame*3); // Récupère les pixels de la frame
      allocation_tableau(ImgOut, OCTET, tailleFrame*3); // Conserve la mosaïque créée à partir de la frame

      // On récupère les pixels de la frame courante que l'on écrit dans ImgIn
      for (int x = 0; x < frame.rows; x++) {
        for (int y = 0; y < frame.cols; y++) {
            Vec3b pixel = frame.at<Vec3b>(x,y);
            // Apparemment c'est au format BGR et non RGB
            uchar blue = pixel[0];
            uchar green = pixel[1];
            uchar red = pixel[2];

            ImgIn[x*frame_width*3 + y*3+0] = red ;
            ImgIn[x*frame_width*3 + y*3+1] = green;
            ImgIn[x*frame_width*3 + y*3+2] = blue;
        }
      }

    

      Mat frameTraite;
      frameTraite.create(frame_height, frame_width, CV_8UC3);
      for (int x = 1; x < frame.rows - 1; x++) {
        for (int y = 1; y < frame.cols - 1; y++) {

            ImgOut[x*frame_width*3 + y*3+0] = (ImgIn[x*frame_width*3 + y*3+0] + ImgIn[(x+1)*frame_width*3 + y*3+0] + ImgIn[(x-1)*frame_width*3 + y*3+0] + ImgIn[(x+1)*frame_width*3 + (y+1)*3+0] + ImgIn[(x-1)*frame_width*3 + (y-1)*3+0] + ImgIn[(x)*frame_width*3 + (y+1)*3+0] + ImgIn[(x+1)*frame_width*3 + (y-1)*3+0] + ImgIn[(x-1)*frame_width*3 + (y+1)*3+0] + ImgIn[(x)*frame_width*3 + (y-1)*3+0])/9;
            ImgOut[x*frame_width*3 + y*3+1] = (ImgIn[x*frame_width*3 + y*3+1] + ImgIn[(x+1)*frame_width*3 + y*3+1] + ImgIn[(x-1)*frame_width*3 + y*3+1] + ImgIn[(x+1)*frame_width*3 + (y+1)*3+1] + ImgIn[(x-1)*frame_width*3 + (y-1)*3+1] + ImgIn[(x)*frame_width*3 + (y+1)*3+1] + ImgIn[(x+1)*frame_width*3 + (y-1)*3+1] + ImgIn[(x-1)*frame_width*3 + (y+1)*3+1] + ImgIn[(x)*frame_width*3 + (y-1)*3+1])/9;
            ImgOut[x*frame_width*3 + y*3+2] = (ImgIn[x*frame_width*3 + y*3+2] + ImgIn[(x+1)*frame_width*3 + y*3+2] + ImgIn[(x-1)*frame_width*3 + y*3+2] + ImgIn[(x+1)*frame_width*3 + (y+1)*3+2] + ImgIn[(x-1)*frame_width*3 + (y-1)*3+2] + ImgIn[(x)*frame_width*3 + (y+1)*3+2] + ImgIn[(x+1)*frame_width*3 + (y-1)*3+2] + ImgIn[(x-1)*frame_width*3 + (y+1)*3+2] + ImgIn[(x)*frame_width*3 + (y-1)*3+2])/9;
        }
      }
   

      // // Ecrire l'image mosaique dans la nouvelle frame
      for (int n = 0 ; n < frame_height ; n++){
        for (int m = 0 ; m < frame_width ; m++){
          // Attention : Dans les frames on écrit au format BGR et non RGB
          frameTraite.at<Vec3b>(n,m) = Vec3b((uchar)ImgOut[n*frame_width*3 + m*3+2], (uchar)ImgOut[n*frame_width*3 + m*3+1], (uchar)ImgOut[n*frame_width*3 + m*3+0]);
        }
      }

      videoTraite.write(frameTraite);
      //imshow("Frame", frameTraite);
      cv::imwrite("frame_output.jpg", frameTraite);
      std::cout << "Une nouvelle frame produite (" << nbFrame << ")\n";
      nbFrame++;

      // Appuyer sur 'q' pour arrêter
      if (waitKey(10) == 'q'){
        break;
      }
  }

  std::cout << "Un total de " << nbFrame << " frames\n";
  cap.release();
  videoTraite.release();
  destroyAllWindows();


  return 0;
}
