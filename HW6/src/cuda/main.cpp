#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <algorithm>
#include "helper.h"
#include "bmpfuncs.h"
#include "Conv.h"
#include "CycleTimer.h"

void usage(const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -i  --iter <INT>       Use specified interation (>=256)\n");
    printf("  -v  --view <INT>       Use specified view settings (1 or 2)\n");
    printf("  -g  --gpu-only <INT>   Only run GPU or not (1 or 0)\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char **argv)
{
   // Rows and columns in the input image
   int imageHeight;
   int imageWidth;

   char *inputFile = "input.bmp";
   const char *outputFile = "output_cuda.bmp";
   char *filterFile = "filter1.csv";

   // parse commandline options ////////////////////////////////////////////
   int opt;
   static struct option long_options[] = {
       {"filter", 1, 0, 'f'},
       {"input", 1, 0, 'i'},
       {"help", 0, 0, '?'},
       {0, 0, 0, 0}};

   while ((opt = getopt_long(argc, argv, "i:f:?", long_options, NULL)) != EOF)
   {

      switch (opt)
      {
      case 'i':
      {
         inputFile = optarg;

         break;
      }
      case 'f':
      {
         int idx = atoi(optarg);
         if (idx == 2)
            filterFile = "filter2.csv";
         else if (idx == 3)
            filterFile = "filter3.csv";

         break;
      }
      case '?':
      default:
         usage(argv[0]);
         return 1;
      }
   }
   // end parsing of commandline options

   // read filter data
   int filterWidth;
   float *filter = readFilter(filterFile, &filterWidth);

   // Homegrown function to read a BMP from file
   float *inputImage = readImage(inputFile, &imageWidth, &imageHeight);
   // Size of the input and output images on the host
   int dataSize = imageHeight * imageWidth * sizeof(float);
   // Output image on the host
   float *outputImage = (float *)malloc(dataSize);
   double minRef = 0;
   double start_time, end_time;
   memset(outputImage, 0, dataSize);
   start_time = currentSeconds();
   hostFE(filterWidth, filter, imageHeight, imageWidth, inputImage, outputImage);
   end_time = currentSeconds();

   printf("[Conv]:\t\t[%.3f] ms\n", (end_time - start_time) * 1000);
   // Run the host to execute the kernel

   // Write the output image to file
   storeImage(outputImage, outputFile, imageHeight, imageWidth, inputFile);

   return 0;
}

