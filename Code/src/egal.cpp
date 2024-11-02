#include <stdio.h>
#include "image_ppm.h"
#include <iostream>
#include <fstream>
using namespace std;

int inverseT(float T[], float val) {
    for (int i = 0; i < 256; i++) {
        if (T[i] >= val) {
            return i;
        }
    }
    return 255; 
}

int main(int argc, char* argv[]){

    char cNomImgLue1[250], cNomImgLue2[250], cNomImgEcrite[250];
    int nH, nW, nTaille, val;
    float nvGris1[256][2], Fx1[256][2], nvGris2[256][2], Fx2[256][2];

    for (int i=0; i < 256; i++){
        nvGris1[i][0] = i;
        nvGris1[i][1] = 0;
        Fx1[i][0] = i;
        Fx1[i][1] = 0;
        nvGris2[i][0] = i;
        nvGris2[i][1] = 0;
        Fx2[i][0] = i;
        Fx2[i][1] = 0;
    }

    OCTET *ImgIn1, *ImgIn2, *ImgOut, *ImgEq;

    sscanf (argv[1],"%s",cNomImgLue1) ;
    sscanf (argv[2],"%s",cNomImgLue2) ;
    sscanf (argv[3],"%s",cNomImgEcrite);

    lire_nb_lignes_colonnes_image_pgm(cNomImgLue1, &nH, &nW);
    nTaille = nH * nW;

    allocation_tableau(ImgIn1, OCTET, nTaille);
    lire_image_pgm(cNomImgLue1, ImgIn1, nH * nW);
    allocation_tableau(ImgIn2, OCTET, nTaille);
    lire_image_pgm(cNomImgLue2, ImgIn2, nH * nW);
    allocation_tableau(ImgOut, OCTET, nTaille);
    allocation_tableau(ImgEq, OCTET, nTaille);


    for (int i=0; i < nH; i++){
        for (int j=0; j < nW; j++){
            nvGris1[ImgIn1[i*nW+j]][1] += 1;
        }
    }

    for(int i =0; i< 256; i++){
        nvGris1[i][1] = nvGris1[i][1] / nTaille;
    }

    Fx1[0][1] = nvGris1[0][1];

    for(int i =1; i< 256; i++){
        Fx1[i][1] = Fx1[i-1][1] + nvGris1[i][1];
    }

    /////////////////////////////////////////////////

    for (int i=0; i < nH; i++){
        for (int j=0; j < nW; j++){
            nvGris2[ImgIn2[i*nW+j]][1] += 1;
        }
    }

    for(int i =0; i< 256; i++){
        nvGris2[i][1] = nvGris2[i][1] / nTaille;
    }

    Fx2[0][1] = nvGris2[0][1];

    for(int i =1; i< 256; i++){
        Fx2[i][1] = Fx2[i-1][1] + nvGris2[i][1];
    }

    //////////////////////////////////////////////////

    // for (int i=0; i < nH; i++){
    //     for (int j=0; j < nW; j++){
    //         ImgEq[i*nW+j]= Fx1[ImgIn1[i*nW+j]][1] * 255;
    //     }
    // }

    // for (int i=0; i < nH; i++){
    //     for (int j=0; j < nW; j++){
    //         for(int x = 0; x<256; x++){
    //             if(Fx1[x][1] >= ImgEq[i*nW+j]/255){
    //                 ImgOut[i*nW+j] = x;
    //             }
    //             else {
    //                 ImgOut[i*nW+j] = 255;
    //             }
    //         }
    //     }
    // }



    for (int i=0; i < nH; i++){
        for (int j=0; j < nW; j++){
            ImgEq[i*nW+j]= Fx1[ImgIn1[i*nW+j]][1] * 255;
        }
    }



    for (int i=0; i < nH; i++){
        for (int j=0; j < nW; j++){
                ImgOut[i*nW+j] = inverseT(Fx1[1], ImgEq[i*nW+j]);
        }
    }

    

    // ofstream myfile;
    // myfile.open("distrib.dat");
    // for(int i=0; i < 256; i++){
    //     myfile << nvGris[i][0];
    //     myfile << " ";
    //     myfile << nvGris[i][1];
    //     myfile << "\n";
    // }
    // myfile.close();
    // myfile.open("FaLena.dat");
    // for(int i=0; i < 256; i++){
    //     myfile << Fx[i][0];
    //     myfile << " ";
    //     myfile << Fx[i][1];
    //     myfile << "\n";
    // }
    // myfile.close();


    ecrire_image_pgm(cNomImgEcrite, ImgOut,  nH, nW);
    free(ImgIn1); free(ImgIn2);
}