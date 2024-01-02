#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>

constexpr bool CV_SUCCESS = true;

#define CHECK_CU(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason : %s\n", error, cudaGetErrorName(error)); \
        exit(-10 * error);\
    } \
} \

#define CHECK_CV(call) { \
    const bool error = call; \
    if (error != CV_SUCCESS) {\
        printf("Error : %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason\n", error); \
        exit(-10 * error);\
    } \
} \

__constant__ float GAUSSIAN_FILTER_SUM = 256.0f;
__constant__ float GAUSSIAN_FILTER[5][5] = {
    {1,  4,  6,  4,  1},
    {4, 16, 24, 16,  4},
    {6, 24, 36, 24,  6},
    {4, 16, 24, 16,  4},
    {1,  4,  6,  4,  1}
};

__global__ void GaussianTextureKernel(uint8_t* output, const cudaTextureObject_t texObj, int width, int height) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

      float3 res = make_float3(0.0f, 0.0f, 0.0f);

      for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
          int offsetX = x + j;
          int offsetY = y + i;

          if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
            float3 pixel = make_float3(
              tex2D<uint8_t>(texObj, offsetX * 3, offsetY),
              tex2D<uint8_t>(texObj, offsetX * 3 + 1, offsetY),
              tex2D<uint8_t>(texObj, offsetX * 3 + 2, offsetY)
            );

            res.x += pixel.x * GAUSSIAN_FILTER[i + 2][j + 2];
            res.y += pixel.y * GAUSSIAN_FILTER[i + 2][j + 2];
            res.z += pixel.z * GAUSSIAN_FILTER[i + 2][j + 2];
          }
        }
      }

      output[y * (width * 3) + (x * 3)] = res.x / GAUSSIAN_FILTER_SUM;
      output[y * (width * 3) + (x * 3) + 1] = res.y / GAUSSIAN_FILTER_SUM;
      output[y * (width * 3) + (x * 3) + 2] = res.z / GAUSSIAN_FILTER_SUM;
    }
}

void initTexture(cudaTextureObject_t& texObj, uint8_t* d_img, cv::Mat& _img, int w, int h) {
    size_t pitch;
    CHECK_CU(cudaMallocPitch(&d_img, &pitch, _img.step1(), h));
    CHECK_CU(cudaMemcpy2D(
        d_img,
        pitch,
        _img.data,
        _img.step1(),
        w * 3 * sizeof(uint8_t),
        h,
        cudaMemcpyHostToDevice));
    
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uint8_t>();
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = d_img;
    texRes.res.pitch2D.desc = desc;
    texRes.res.pitch2D.width = 3 * w;
    texRes.res.pitch2D.height = h;
    texRes.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    CHECK_CU(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}

void run_1_task(std::string inputPath, std::string outputPath, uint8_t radius) {
    try {
        const std::string filename = inputPath;
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        

        if (image.empty()) {
            printf("Cannot read image file: %s\n", filename.c_str());
            return;
        }

        cudaTextureObject_t texObj;
        uint8_t* d_img;
        uint8_t* d_output;
        uint8_t* gpuRef = new uint8_t[image.cols * image.rows * sizeof(uint8_t) * 3];

        initTexture(texObj, d_img, image, image.cols, image.rows);

        // Allocate result of transformation in device memory
        CHECK_CU(cudaMalloc((void **) &d_output, image.cols * image.rows * sizeof(uint8_t) * 3));


        // Invoke kernel
        dim3 dimBlock(16, 16);
        dim3 dimGrid(
            (image.cols + dimBlock.x - 1) / dimBlock.x,
            (image.rows + dimBlock.y - 1) / dimBlock.y
        );

        printf("Kernel Dimension :\n   Block size : %i , %i \n    Grid size : %i , %i",
            dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y
        );

        GaussianTextureKernel <<< dimGrid, dimBlock >>> (d_output, texObj, image.cols, image.rows);

        CHECK_CU(cudaMemcpy(gpuRef, d_output, image.cols * image.rows * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost));
        cv::Mat imageOut = cv::Mat(image.rows, image.cols, CV_8UC3, gpuRef);
        CHECK_CV(imwrite(outputPath, imageOut));

    
        delete[] gpuRef;
        CHECK_CU(cudaFree(d_output))
        CHECK_CU(cudaFree(d_img))
        CHECK_CU(cudaDestroyTextureObject(texObj))
    }
    catch (cv::Exception ex) {
        std::cerr << ex.what() << std::endl;
    }
}

std::vector<std::vector<float>> generateGaussianMatrixFilter(uint8_t radius) {
   std::vector<std::vector<float>> gaussianFilter(radius, std::vector<float>(radius));
   double sigma = 1;
   double mean = radius / 2;
   double sum = 0.0;

   for (int i = 0; i < radius; ++i) {
       for (int j = 0; j < radius; ++j) {
        // Разбиваем на слагаемые и множители
        double arg_x = ((i - mean) / sigma) * ((i - mean) / sigma);
        double arg_y = ((j - mean) / sigma) * ((j - mean) / sigma);
        double exponential_term = exp(-0.5 * (arg_x + arg_y));

        double normalization_constant = 2 * M_PI * sigma * sigma;

        gaussianFilter[i][j] = exponential_term / normalization_constant;
        sum += gaussianFilter[i][j];
       }
   }

   
   for (int x = 0; x < radius; ++x) {
       for (int y = 0; y < radius; ++y) {
           gaussianFilter[x][y] /= sum;
       }
   }

    std::cout << "матрица весов: \n";
    for (auto row: gaussianFilter) {
        for (auto item: row ) {
            std::cout <<  item << ' ';
        }
        std::cout << '\n';
    }

   return gaussianFilter;
}


int main() {


    run_1_task("images/simple_kvashino.jpg", "images/out.jpg", 3);

    return 0;
}

