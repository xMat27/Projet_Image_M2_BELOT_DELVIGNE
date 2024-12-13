// test_couleur.cpp : Seuille une image en niveau de gris

#include <stdio.h>
#include "image_ppm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
  char cNomVideoLue[250], cNomVideoTraite[250];
  float nvR[256][2], FxR[256][2], nvG[256][2], FxG[256][2], nvB[256][2], FxB[256][2];

  
  
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

    for (int i=0; i < 256; i++){
        nvR[i][0] = i;
        nvR[i][1] = 0;
        FxR[i][0] = i;
        FxR[i][1] = 0;
        nvG[i][0] = i;
        nvG[i][1] = 0;
        FxG[i][0] = i;
        FxG[i][1] = 0;
        nvB[i][0] = i;
        nvB[i][1] = 0;
        FxB[i][0] = i;
        FxB[i][1] = 0;
    }
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

            nvR[ImgIn[x*frame_width*3 + y*3+0]][1] += 1;
            nvG[ImgIn[x*frame_width*3 + y*3+1]][1] += 1;
            nvB[ImgIn[x*frame_width*3 + y*3+2]][1] += 1;
        }
      }

    for(int i =0; i< 256; i++){
        nvR[i][1] = nvR[i][1] / tailleFrame;
        nvG[i][1] = nvG[i][1] / tailleFrame;
        nvB[i][1] = nvB[i][1] / tailleFrame;
    }

    FxR[0][1] = nvR[0][1];
    FxG[0][1] = nvG[0][1];
    FxB[0][1] = nvB[0][1];

    for(int i =1; i< 256; i++){
        FxR[i][1] = FxR[i-1][1] + nvR[i][1];
        FxG[i][1] = FxG[i-1][1] + nvG[i][1];
        FxB[i][1] = FxB[i-1][1] + nvB[i][1];
    }

    

      Mat frameTraite;
      frameTraite.create(frame_height, frame_width, CV_8UC3);
      for (int x = 1; x < frame.rows - 1; x++) {
        for (int y = 1; y < frame.cols - 1; y++) {
            ImgOut[x*frame_width*3 + y*3+0] = FxR[ImgIn[x*frame_width*3 + y*3+0]][1] * 255;
            ImgOut[x*frame_width*3 + y*3+1] = FxG[ImgIn[x*frame_width*3 + y*3+1]][1] * 255;
            ImgOut[x*frame_width*3 + y*3+2] = FxB[ImgIn[x*frame_width*3 + y*3+2]][1] * 255;
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
      imwrite("frame_egal.jpg", frameTraite);
      imwrite("frame_originale.jpg", frame);
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
