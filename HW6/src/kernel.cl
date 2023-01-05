__kernel void convolution(int filterWidth, __global float *filter, int imageHeight, int imageWidth, __global float *inputImage, __global float *outputImage)
{
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum = 0; // Reset sum for new source pixel

    // get index
    int row = get_global_id(1), col = get_global_id(0);  

    // compute boundary
    int k_start = row - halffilterSize >= 0 ? -halffilterSize : 0, k_end = halffilterSize + row < imageHeight ? halffilterSize : halffilterSize + row - imageHeight - 1;
    int l_start = col - halffilterSize >= 0 ? -halffilterSize : 0, l_end = halffilterSize + col < imageWidth ? halffilterSize : halffilterSize + col - imageWidth - 1;

    // apply filter
    // if filter is 3*3, unroll (because testcases' filters are all 3*3 after cut)
    if (k_end - k_start + 1 == 3 && l_end - l_start + 1 == 3)
    {
        sum += inputImage[(row + k_start) * imageWidth + col + l_start] * filter[(k_start + halffilterSize) * filterWidth + halffilterSize + l_start];
        sum += inputImage[(row + k_start) * imageWidth + col + l_start + 1] * filter[(k_start + halffilterSize) * filterWidth + halffilterSize + l_start + 1];
        sum += inputImage[(row + k_start) * imageWidth + col + l_start + 2] * filter[(k_start + halffilterSize) * filterWidth + halffilterSize + l_start + 2];

        sum += inputImage[(row + k_start + 1) * imageWidth + col + l_start] * filter[(k_start + 1 + halffilterSize) * filterWidth + halffilterSize + l_start];
        sum += inputImage[(row + k_start + 1) * imageWidth + col + l_start + 1] * filter[(k_start + 1 + halffilterSize) * filterWidth + halffilterSize + l_start + 1];
        sum += inputImage[(row + k_start + 1) * imageWidth + col + l_start + 2] * filter[(k_start + 1 + halffilterSize) * filterWidth + halffilterSize + l_start + 2];

        sum += inputImage[(row + k_start + 2) * imageWidth + col + l_start] * filter[(k_start + 2 + halffilterSize) * filterWidth + halffilterSize + l_start];
        sum += inputImage[(row + k_start + 2) * imageWidth + col + l_start + 1] * filter[(k_start + 2 + halffilterSize) * filterWidth + halffilterSize + l_start + 1];
        sum += inputImage[(row + k_start + 2) * imageWidth + col + l_start + 2] * filter[(k_start + 2 + halffilterSize) * filterWidth + halffilterSize + l_start + 2];
    }
    // normal way
    else 
        for (int k = k_start; k <= k_end; ++k)
        {
            int idxImage = (row + k) * imageWidth + col, idxFilter = (k + halffilterSize) * filterWidth + halffilterSize;
            for (int l = l_start; l <= l_end; ++l) 
                sum += inputImage[idxImage + l] * filter[idxFilter + l];        
        }

    // assign output
    outputImage[row * imageWidth + col] = sum;
}
