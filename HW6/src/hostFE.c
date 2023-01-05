#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status; // store error code
    int filterSize = filterWidth * filterWidth;

    // add by myself
    // context, program, device are already prepared by TA
    // create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, &status);
    CHECK(status, "clCreateCommandQueue");
    
    // create input image buffer
    size_t image_size = sizeof(float) * imageHeight * imageWidth;
    cl_mem d_input = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_size, NULL, &status);
    CHECK(status, "clCreateBuffer_image");

    // create output image buffer
    cl_mem d_output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, &status);
    CHECK(status, "clCreateBuffer_output");

    // transfer image
    status = clEnqueueWriteBuffer(commandQueue, d_input, CL_TRUE, 0, image_size, (void*) inputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer_image");

    // cut filter
    int filterCutSzie = 0;
    int canCut = 1;
    do
    {
        int checkEnd = (filterWidth - filterCutSzie - 1);
        // up
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * filterCutSzie + i] != 0)
                canCut = 0;

        // down
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * checkEnd + i] != 0)
                canCut = 0;

        // left
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * i + filterCutSzie] != 0)
                canCut = 0;

        // right
        for (int i = 0; i < filterWidth; ++i)
            if (filter[filterWidth * i + checkEnd] != 0)
                canCut = 0;
    } while (canCut && ++filterCutSzie && filterCutSzie <= filterWidth / 2);

    int cuttedFilterWidth = filterWidth - filterCutSzie * 2;
    float *cuttedFilter = (float*) malloc(sizeof(float) * cuttedFilterWidth * cuttedFilterWidth);
    size_t cuttedFilterSize = sizeof(float) * cuttedFilterWidth * cuttedFilterWidth;
    for (int i = 0 ; i < cuttedFilterWidth; ++i)
        for (int j = 0; j < cuttedFilterWidth; ++j)
            cuttedFilter[cuttedFilterWidth * i + j] = filter[filterWidth * (i + filterCutSzie) + (j + filterCutSzie)];

    // create input filter buffer
    cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY, cuttedFilterSize, NULL, &status);
    CHECK(status, "clCreateBuffer_filter");

    // transfer filter
    status = clEnqueueWriteBuffer(commandQueue, d_filter, CL_TRUE, 0, cuttedFilterSize, (void*) cuttedFilter, 0, NULL, NULL);
    CHECK(status, "clEnqueueWriteBuffer_filter");

    // create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);
    CHECK(status, "clCreateKernel");

    // set arg.
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*) &cuttedFilterWidth);
    CHECK(status, "clSetKernelArg_0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &d_filter);
    CHECK(status, "clSetKernelArg_1");
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*) &imageHeight);
    CHECK(status, "clSetKernelArg_2");
    status = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*) &imageWidth);
    CHECK(status, "clSetKernelArg_3");
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) &d_input);
    CHECK(status, "clSetKernelArg_4");
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*) &d_output);
    CHECK(status, "clSetKernelArg_5");

    // set work group and work item size
    size_t localws[2] = {8, 8};
    size_t globalws[2] = {imageWidth, imageHeight};

    // execute kernel
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);
    CHECK(status, "clEnqueueNDRandgeKernel");

    // copy results from device to host
    status = clEnqueueReadBuffer(commandQueue, d_output, CL_TRUE, 0, image_size, (void*) outputImage, 0, NULL, NULL);
    CHECK(status, "clEnqueueReadBuffer");

    // release
    status = clReleaseCommandQueue(commandQueue);
    CHECK(status, "clReleaseCommandQueue");
    status = clReleaseMemObject(d_input);
    CHECK(status, "clReleaseMemObject_input");
    status = clReleaseMemObject(d_output);
    CHECK(status, "clReleaseMemObject_output");
    status = clReleaseMemObject(d_filter);
    CHECK(status, "clReleaseMemObject_filter");
    status = clReleaseKernel(kernel);
    CHECK(status, "clReleaseKernel");
}