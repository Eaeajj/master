#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>

constexpr bool CV_SUCCESS = true;
constexpr uint16_t COLOR_SIZE = sizeof(uint8_t) * 4; // uchar4
constexpr uint16_t RGB_SIZE = sizeof(uint8_t) * 3;

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


// --------------------------------------------------------- 1 task ------------------------------------------------------


void init_texture(cudaTextureObject_t& tex_obj, uint8_t* d_img, cv::Mat& _img, int w, int h) {
    size_t pitch;
    CHECK_CU(cudaMallocPitch(&d_img, &pitch, _img.step1(), h));
    CHECK_CU(cudaMemcpy2D(
        d_img,
        pitch,
        _img.data,
        _img.step1(),
        w * RGB_SIZE,
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
        uint8_t* gpu_ref = new uint8_t[image.cols * image.rows * RGB_SIZE];

        init_texture(tex_obj, d_img, image, image.cols, image.rows);

        // Allocate result of transformation in device memory
        CHECK_CU(cudaMalloc((void **) &d_output, image.cols * image.rows * RGB_SIZE));


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

        CHECK_CU(cudaMemcpy(gpu_ref, d_output, image.cols * image.rows * RGB_SIZE, cudaMemcpyDeviceToHost));
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

// --------------------------------------------------------- 2 task ------------------------------------------------------
void init_texture_u4(cudaTextureObject_t& tex_obj, uchar4* d_img, cv::Mat& _img) {
    size_t pitch;
    int w = _img.cols;
    int h = _img.rows;

    // Allocate memory for uchar4 with cudaMallocPitch
    CHECK_CU(cudaMallocPitch(&d_img, &pitch, w, h));

    // Copy the image data to the device memory
    CHECK_CU(cudaMemcpy2D(
        d_img,
        pitch,
        _img.data,
        _img.step1(),
        w * COLOR_SIZE,
        h,
        cudaMemcpyHostToDevice));

    // Set up the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypePitch2D;
    texRes.res.pitch2D.devPtr = d_img;
    texRes.res.pitch2D.desc = desc;
    texRes.res.pitch2D.width = w;
    texRes.res.pitch2D.height = h;
    texRes.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeElementType;

    // Create the texture object
    CHECK_CU(cudaCreateTextureObject(&tex_obj, &texRes, &texDescr, NULL));
}

__device__ uchar4 operator*(uchar4 a, float b) {
  return reinterpret_cast<uchar4>(a * b);
}

__device__ uchar get(uchar4 a, uchar i) {
  switch (i) {
    case 0: return a.x;
    case 1: return a.y;
    case 2: return a.z;
    case 3: return a.w;
    default: return 0;
  }
}

__device__ void set(uchar4* a, uchar i, uchar value) {
  switch (i) {
    case 0: a->x = value; break;
    case 1: a->y = value; break;
    case 2: a->z = value; break;
    case 3: a->w = value; break;
    default: break;
  }
}
__device__ uchar4 avg_sum(uchar4 a, uchar4 b, uchar4 c, uchar4 d) {
  uchar4 result;
  for (uchar i = 0; i < 3; i++) {
    uchar res = (get(a, i) + get(b, i) + get(c, i) + get(d, i)) / 4;
    set(&result, i, res);
  }
  return result;
}


__global__ void bilinear_kernel(uchar4* output, const cudaTextureObject_t texObj, size_t width, size_t height, size_t scale_coeff) {
    // Calculate the index in the new image
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the index is within the boundaries of the new image
    if (x < width && y < height) {
        // Calculate the coordinates in the original image using scaled coordinates
        float fx = x / scale_coeff;
        float fy = y / scale_coeff;

        // Interpolate between four nearest neighbors in the original image
        float fx0 = floor(fx);
        float fx1 = ceil(fx);
        float fy0 = floor(fy);
        float fy1 = ceil(fy);

        uchar4 px00 = tex2D<uchar4>(texObj, fx0, fy0);
        uchar4 px10 = tex2D<uchar4>(texObj, fx1, fy0);
        uchar4 px01 = tex2D<uchar4>(texObj, fx0, fy1);
        uchar4 px11 = tex2D<uchar4>(texObj, fx1, fy1);

        // Calculate the weights for each neighbor based on subpixel interpolation
        float w00 = (fx1 - fx) * (fy1 - fy);
        float w10 = (fx - fx0) * (fy1 - fy);
        float w01 = (fx1 - fx) * (fy - fy0);
        float w11 = (fx - fx0) * (fy - fy0);

        // Calculate the interpolated pixel color
        uchar4 interpolatedColor = avg_sum(
            px00 * w00,
            px10 * w10,
            px01 * w01,
            px11 * w11
        );

        // Write the interpolated color to the output image
        output[y * width + x] = interpolatedColor;
    }
}

void run_2_task(std::string input_path, std::string output_path) {
    try {
        cudaDeviceSynchronize();
        const std::string filename = input_path;
        cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
        std::cout << "cols: " << image.cols << " rows: " << image.rows << " s: " << image.size << "\n";
        if (image.empty()) {
            printf("Cannot read image file: %s\n", filename.c_str());
            return;
        }

        cudaTextureObject_t tex_obj;
        uchar4* d_img;
        uchar4* d_output;

        const auto scale_coeff = 2;
        const auto new_w = image.cols * scale_coeff;
        const auto new_h = image.rows * scale_coeff;
        

        uchar4* gpu_ref = new uchar4[new_w * new_h * COLOR_SIZE];
        init_texture_u4(tex_obj, d_img, image);

        // Allocate result of transformation in device memory
        CHECK_CU(cudaMalloc((void **)&d_output, new_w * new_h * COLOR_SIZE));

        // Invoke kernel
        dim3 dim_block(16, 16);
        dim3 dim_grid(
            (new_w + dim_block.x) / dim_block.x,
            (new_h + dim_block.y) / dim_block.y
        );

        printf("Kernel Dimension :\n   Block size : %i , %i \n    Grid size : %i , %i\n",
               dim_block.x, dim_block.y, dim_grid.x, dim_grid.y);

        bilinear_kernel<<<dim_grid, dim_block>>>(d_output, tex_obj, new_w, new_h, scale_coeff);

        // CHECK_CU(cudaMemcpy(gpu_ref, d_output, new_w * new_h * RGB_SIZE, cudaMemcpyDeviceToHost));
        cv::Mat img_out = cv::Mat(new_h, new_w, CV_8UC3); // 
        cudaMemcpy(img_out.data, d_output, new_h * new_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
        std::cout << "cols: " << img_out.cols << " rows: " << img_out.rows << " s: " << img_out.size << "\n";
        CHECK_CV(imwrite(output_path, img_out));

        delete[] gpu_ref;
        CHECK_CU(cudaFree(d_output));
        CHECK_CU(cudaDestroyTextureObject(tex_obj));
        CHECK_CU(cudaFree(d_img));
        cudaDeviceSynchronize();
        cudaDeviceReset();
    } catch (cv::Exception ex) {
        std::cerr << ex.what() << std::endl;
    }
}



int main() {
    test_1_task();

    // run_2_task("images/mountains-small.jpg", "images/out.jpg");
    return 0;
}

