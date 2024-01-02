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

// ---------------------------- task 1 --------------------------------

std::vector<std::vector<float>> generate_gaussian_matrix(uint8_t radius) {
   std::vector<std::vector<float>> gaussian_filter(radius, std::vector<float>(radius));
   double sigma = 1;
   double mean = radius / 2;
   double sum = 0.0;

   for (int i = 0; i < radius; ++i) {
       for (int j = 0; j < radius; ++j) {
        double arg_x = ((i - mean) / sigma) * ((i - mean) / sigma);
        double arg_y = ((j - mean) / sigma) * ((j - mean) / sigma);
        double exponential_term = exp(-0.5 * (arg_x + arg_y));

        double normalization_constant = 2 * M_PI * sigma * sigma;

        gaussian_filter[i][j] = exponential_term / normalization_constant;
        sum += gaussian_filter[i][j];
       }
   }

   
   for (int x = 0; x < radius; ++x) {
       for (int y = 0; y < radius; ++y) {
           gaussian_filter[x][y] /= sum;
       }
   }

    std::cout << "матрица весов: \n";
    for (auto row: gaussian_filter) {
        for (auto item: row ) {
            std::cout <<  item << ' ';
        }
        std::cout << '\n';
    }

   return gaussian_filter;
}

__global__ void gaussian_texture_kernel(uint8_t* output, const cudaTextureObject_t tex_obj, int width, int height, float* d_gaussian_matrix, uint8_t diametr) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint8_t r = diametr / 2; // radius == 3 => -2, -1, 0, 1, 2
    if (x < width && y < height) {

      float3 res = make_float3(0.0f, 0.0f, 0.0f);

      for (int i = -r; i <= r; ++i) {
        for (int j = -r; j <= r; ++j) {
          int offsetX = x + j;
          int offsetY = y + i;

          if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
            float3 pixel = make_float3(
              tex2D<uint8_t>(tex_obj, offsetX * 3, offsetY),
              tex2D<uint8_t>(tex_obj, offsetX * 3 + 1, offsetY),
              tex2D<uint8_t>(tex_obj, offsetX * 3 + 2, offsetY)
            ); // color

            auto curr_multiplier = d_gaussian_matrix[(i + r) * r + (j + r)];
            res.x += pixel.x * curr_multiplier;
            res.y += pixel.y * curr_multiplier;
            res.z += pixel.z * curr_multiplier;
          }
        }
      }

      output[y * (width * 3) + (x * 3)] = res.x;
      output[y * (width * 3) + (x * 3) + 1] = res.y;
      output[y * (width * 3) + (x * 3) + 2] = res.z;
    }
}

void init_texture(cudaTextureObject_t& tex_obj, uint8_t* d_img, cv::Mat& _img, int w, int h) {
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

    CHECK_CU(cudaCreateTextureObject(&tex_obj, &texRes, &texDescr, NULL));
}

void run_1_task(std::string input_path, std::string output_path, uint8_t diametr) {
    try {
        cudaDeviceSynchronize();
        const std::string filename = input_path;
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        

        if (image.empty()) {
            printf("Cannot read image file: %s\n", filename.c_str());
            return;
        }
        auto host_gaussian_filter = generate_gaussian_matrix(diametr);


        std::vector<float> flat_filter;
        for (const auto& row : host_gaussian_filter) {
            flat_filter.insert(flat_filter.end(), row.begin(), row.end());
        }

        float* d_gaussianFilter;
        CHECK_CU(cudaMalloc((void**)&d_gaussianFilter, flat_filter.size() * sizeof(float)));
        CHECK_CU(cudaMemcpy(d_gaussianFilter, flat_filter.data(), flat_filter.size() * sizeof(float), cudaMemcpyHostToDevice));

        cudaTextureObject_t tex_obj;
        uint8_t* d_img;
        uint8_t* d_output;
        uint8_t* gpu_ref = new uint8_t[image.cols * image.rows * sizeof(uint8_t) * 3];

        init_texture(tex_obj, d_img, image, image.cols, image.rows);

        // Allocate result of transformation in device memory
        CHECK_CU(cudaMalloc((void **) &d_output, image.cols * image.rows * sizeof(uint8_t) * 3));


        // Invoke kernel
        dim3 dim_block(16, 16);
        dim3 dim_grid(
            (image.cols + dim_block.x - 1) / dim_block.x,
            (image.rows + dim_block.y - 1) / dim_block.y
        );

        printf("Kernel Dimension :\n   Block size : %i , %i \n    Grid size : %i , %i\n",
            dim_block.x, dim_block.y, dim_grid.x, dim_grid.y
        );

        gaussian_texture_kernel <<< dim_grid, dim_block >>> (d_output, tex_obj, image.cols, image.rows, d_gaussianFilter, diametr);

        CHECK_CU(cudaMemcpy(gpu_ref, d_output, image.cols * image.rows * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost));
        cv::Mat image_out = cv::Mat(image.rows, image.cols, CV_8UC3, gpu_ref);
        CHECK_CV(imwrite(output_path, image_out));

    
        delete[] gpu_ref;
        CHECK_CU(cudaFree(d_output));
        CHECK_CU(cudaDestroyTextureObject(tex_obj));
        CHECK_CU(cudaFree(d_gaussianFilter));
        CHECK_CU(cudaFree(d_img));
        cudaDeviceSynchronize();
        cudaDeviceReset();
    }
    catch (cv::Exception ex) {
        std::cerr << ex.what() << std::endl;
    }
}

void test_1_task() {
    const auto img_path = "images/simple_kvashino.jpg";
    run_1_task(img_path, "images/out3.jpg", 3);
    run_1_task(img_path, "images/out5.jpg", 5);
    run_1_task(img_path, "images/out7.jpg", 7);
    run_1_task(img_path, "images/out9.jpg", 9);
    run_1_task(img_path, "images/out11.jpg", 11);
}

// ---------------------------- task 2 --------------------------------



int main() {
    // test_1_task();

    return 0;
}

