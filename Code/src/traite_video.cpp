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

      // // Création de la mosaïque de la frame courante
      // Mat frameMosa;
      // frameMosa.create(frame_height, frame_width, CV_8UC3);
      // double moyenne_r, moyenne_g, moyenne_b;
      // for(int i=0;i<nb_h;i++){
      //   for(int j=0;j<nb_w;j++){
      //     moyenne_r=0;
      //     moyenne_g=0;
      //     moyenne_b=0;
      //     for (int x = 0; x < taille_imagette; x++) {
      //       for (int y = 0; y < taille_imagette; y++) {
      //         int pixel_index = (i * taille_imagette * frame_width*3) + (x * frame_width*3) + (j * taille_imagette*3) + y*3;
      //         moyenne_r += ImgIn[pixel_index + 0];
      //         moyenne_g += ImgIn[pixel_index + 1];
      //         moyenne_b += ImgIn[pixel_index + 2];
      //       }
      //     }
      //     moyenne_r=moyenne_r/(taille_imagette*taille_imagette);
      //     moyenne_g=moyenne_g/(taille_imagette*taille_imagette);
      //     moyenne_b=moyenne_b/(taille_imagette*taille_imagette);
      //     char* acc;
      //     if (nbFrameIntacte==-1 || compteurFrame < nbFrameIntacte){ // Si l'option n'est pas coché dans l'application, cette condition sera vrai
      //       int indice=0;
      //       acc = (char*)nom[0].c_str();
      //       double stock = abs(moyenne_r-moy_r[0])+abs(moyenne_g-moy_g[0])+abs(moyenne_b-moy_b[0]);
      //       for(int b=1;b<nbImagette;b++){
      //         if(abs(moyenne_r-moy_r[b])+abs(moyenne_g-moy_g[b])+abs(moyenne_b-moy_b[b]) < stock){
      //           stock=abs(moyenne_r-moy_r[b])+abs(moyenne_g-moy_g[b])+abs(moyenne_b-moy_b[b]);
      //           acc=(char*)nom[b].c_str();
      //           indice=b;
      //         }       
      //       }
      //       if (nbFrameIntacte != -1){ // Seulement dans le cas de l'optimisation
      //         if (std::find(alreadyUsed.begin(), alreadyUsed.end(), indice) == alreadyUsed.end()){ // On ajoute si l'élément n'est pas déjà présent
      //           alreadyUsed.push_back(indice);
      //         }
      //       }
      //     }else{ // Seulement si l'option d'optimisation est cochée dans l'application et qu'on a passé les frames intactes. A partir de là on utilise plus que les imagettes déjà utilisées
      //       int indice=alreadyUsed[0];
      //       acc = (char*)nom[alreadyUsed[0]].c_str();
      //       double stock = abs(moyenne_r-moy_r[alreadyUsed[0]])+abs(moyenne_g-moy_g[alreadyUsed[0]])+abs(moyenne_b-moy_b[alreadyUsed[0]]);
      //       for (int a = 1 ; a < alreadyUsed.size() ; a++){
      //         int b = alreadyUsed[a];
      //         if(abs(moyenne_r-moy_r[b])+abs(moyenne_g-moy_g[b])+abs(moyenne_b-moy_b[b]) < stock){
      //           stock=abs(moyenne_r-moy_r[b])+abs(moyenne_g-moy_g[b])+abs(moyenne_b-moy_b[b]);
      //           acc=(char*)nom[b].c_str();
      //           indice=b;
      //         }       
      //       }
      //     }

      //     char* res = new char[strlen(acc) + strlen(repertoireImagette) + 2];
      //     strcpy(res, repertoireImagette);
      //     strcat(res, "/");
      //     strcat(res, acc);


      //     OCTET * Imgacc;
      //     allocation_tableau(Imgacc, OCTET,taille_imagette*taille_imagette*3);
      //     lire_image_ppm(res,Imgacc,taille_imagette*taille_imagette);
      //     for(int z=0;z<taille_imagette;z++){
      //       for(int k=0;k<taille_imagette;k++){
      //         int pixel_index_out = i*taille_imagette*frame_width*3 + z*frame_width*3 + j*taille_imagette*3 + k*3;
      //         int pixel_index_acc = z*taille_imagette*3 + k*3;
      //         ImgOut[pixel_index_out+0] = Imgacc[pixel_index_acc+0];
      //         ImgOut[pixel_index_out+1] = Imgacc[pixel_index_acc+1];
      //         ImgOut[pixel_index_out+2] = Imgacc[pixel_index_acc+2];
      //       }        
      //     }
      //     free(Imgacc);
      //   }
      //}

      Mat frameTraite;
      frameTraite.create(frame_height, frame_width, CV_8UC3);
      for (int x = 0; x < frame.rows; x++) {
        for (int y = 0; y < frame.cols; y++) {

            ImgOut[x*frame_width*3 + y*3+0] = ImgIn[x*frame_width*3 + y*3+0];
            ImgOut[x*frame_width*3 + y*3+1] = ImgIn[x*frame_width*3 + y*3+1];
            ImgOut[x*frame_width*3 + y*3+2] = ImgIn[x*frame_width*3 + y*3+2];
        }
      }

      // if (compteurFrame < nbFrameIntacte){
      //   ++compteurFrame;
      //   cout<<"Nb imagettes jusque là : " << alreadyUsed.size()<<endl; // Affiche le nombre d'imagette qui ont été utilisé jusqu'à la frame courante. A la fin frame intacte, ce sera le nombre final d'imagettes pour les frames suivantes
      // }    

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
