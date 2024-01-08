cmake_minimum_required(VERSION 3.10)

project(MyCudaOpenCVProject LANGUAGES CXX CUDA)

# Устанавливаем стандарт C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Устанавливаем компилятор nvcc для CUDA
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")

# Находим пакет OpenCV
find_package(OpenCV 4 REQUIRED)

# Добавляем исполняемый файл
add_executable(run "src/main.cu")

# Добавляем пути для заголовочных файлов OpenCV
target_include_directories(run PRIVATE ${OpenCV_INCLUDE_DIRS})

# Линкуемся с библиотеками OpenCV
target_link_libraries(run PRIVATE ${OpenCV_LIBS})
