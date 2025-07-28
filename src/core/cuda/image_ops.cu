#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

namespace hybrid_sfm
{
    namespace cuda
    {

        // Custom CUDA kernels for specialized operations

        // Kernel for fast image normalization
        __global__ void normalizeKernel(const uchar *input, float *output,
                                        int width, int height, int channels,
                                        float mean, float stddev)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height)
                return;

            int idx = (y * width + x) * channels;

            for (int c = 0; c < channels; ++c)
            {
                float val = static_cast<float>(input[idx + c]);
                output[idx + c] = (val - mean) / stddev;
            }
        }

        // Kernel for fast histogram computation
        __global__ void histogramKernel(const uchar *input, int *histogram,
                                        int width, int height)
        {
            __shared__ int local_hist[256];

            // Initialize shared memory
            int tid = threadIdx.x + threadIdx.y * blockDim.x;
            if (tid < 256)
            {
                local_hist[tid] = 0;
            }
            __syncthreads();

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height)
            {
                int idx = y * width + x;
                atomicAdd(&local_hist[input[idx]], 1);
            }
            __syncthreads();

            // Write to global memory
            if (tid < 256)
            {
                atomicAdd(&histogram[tid], local_hist[tid]);
            }
        }

        // Kernel for image gradient computation (Sobel)
        __global__ void sobelKernel(const uchar *input, float *grad_x, float *grad_y,
                                    int width, int height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
                return;

            // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
            float gx = -1.0f * input[(y - 1) * width + (x - 1)] + 1.0f * input[(y - 1) * width + (x + 1)] - 2.0f * input[y * width + (x - 1)] + 2.0f * input[y * width + (x + 1)] - 1.0f * input[(y + 1) * width + (x - 1)] + 1.0f * input[(y + 1) * width + (x + 1)];

            // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
            float gy = -1.0f * input[(y - 1) * width + (x - 1)] - 2.0f * input[(y - 1) * width + x] - 1.0f * input[(y - 1) * width + (x + 1)] + 1.0f * input[(y + 1) * width + (x - 1)] + 2.0f * input[(y + 1) * width + x] + 1.0f * input[(y + 1) * width + (x + 1)];

            grad_x[y * width + x] = gx;
            grad_y[y * width + x] = gy;
        }

        // Kernel for non-maximum suppression (useful for feature detection)
        __global__ void nonMaxSuppressionKernel(const float *magnitude, const float *angle,
                                                uchar *output, int width, int height,
                                                float threshold)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
            {
                return;
            }

            int idx = y * width + x;
            float mag = magnitude[idx];

            if (mag < threshold)
            {
                output[idx] = 0;
                return;
            }

            float ang = angle[idx];

            // Quantize angle to 8 directions
            int direction = ((int)(ang / 22.5f + 0.5f)) % 8;

            float mag1, mag2;
            switch (direction)
            {
            case 0:
            case 4: // Horizontal
                mag1 = magnitude[idx - 1];
                mag2 = magnitude[idx + 1];
                break;
            case 1:
            case 5: // Diagonal /
                mag1 = magnitude[idx - width - 1];
                mag2 = magnitude[idx + width + 1];
                break;
            case 2:
            case 6: // Vertical
                mag1 = magnitude[idx - width];
                mag2 = magnitude[idx + width];
                break;
            case 3:
            case 7: // Diagonal \
            mag1 = magnitude[idx - width + 1];
                mag2 = magnitude[idx + width - 1];
                break;
            }

            output[idx] = (mag >= mag1 && mag >= mag2) ? 255 : 0;
        }

        // Wrapper class for CUDA operations
        class CUDAImageOps
        {
        public:
            // Normalize image on GPU
            static cv::cuda::GpuMat normalize(const cv::cuda::GpuMat &input,
                                              float mean = 128.0f, float stddev = 64.0f)
            {
                cv::cuda::GpuMat output(input.size(), CV_32FC(input.channels()));

                dim3 block(16, 16);
                dim3 grid((input.cols + block.x - 1) / block.x,
                          (input.rows + block.y - 1) / block.y);

                normalizeKernel<<<grid, block>>>(
                    input.ptr<uchar>(), output.ptr<float>(),
                    input.cols, input.rows, input.channels(),
                    mean, stddev);

                cudaDeviceSynchronize();
                return output;
            }

            // Compute histogram on GPU
            static cv::Mat computeHistogram(const cv::cuda::GpuMat &input)
            {
                cv::Mat histogram = cv::Mat::zeros(256, 1, CV_32S);

                cv::cuda::GpuMat d_histogram;
                d_histogram.upload(histogram);

                dim3 block(16, 16);
                dim3 grid((input.cols + block.x - 1) / block.x,
                          (input.rows + block.y - 1) / block.y);

                histogramKernel<<<grid, block>>>(
                    input.ptr<uchar>(), d_histogram.ptr<int>(),
                    input.cols, input.rows);

                cudaDeviceSynchronize();
                d_histogram.download(histogram);

                return histogram;
            }

            // Compute image gradients
            static void computeGradients(const cv::cuda::GpuMat &input,
                                         cv::cuda::GpuMat &grad_x,
                                         cv::cuda::GpuMat &grad_y)
            {
                grad_x.create(input.size(), CV_32F);
                grad_y.create(input.size(), CV_32F);

                dim3 block(16, 16);
                dim3 grid((input.cols + block.x - 1) / block.x,
                          (input.rows + block.y - 1) / block.y);

                sobelKernel<<<grid, block>>>(
                    input.ptr<uchar>(), grad_x.ptr<float>(), grad_y.ptr<float>(),
                    input.cols, input.rows);

                cudaDeviceSynchronize();
            }

            // Fast bilateral filter using CUDA
            static cv::cuda::GpuMat bilateralFilter(const cv::cuda::GpuMat &input,
                                                    int d = 9, float sigmaColor = 75.0f,
                                                    float sigmaSpace = 75.0f)
            {
                cv::cuda::GpuMat output;
                cv::cuda::bilateralFilter(input, output, d, sigmaColor, sigmaSpace);
                return output;
            }

            // Multi-scale pyramid with custom interpolation
            static std::vector<cv::cuda::GpuMat> buildPyramidCustom(
                const cv::cuda::GpuMat &input, int levels, float scale = 0.5f)
            {

                std::vector<cv::cuda::GpuMat> pyramid;
                pyramid.reserve(levels);
                pyramid.push_back(input);

                cv::cuda::GpuMat current = input;
                for (int i = 1; i < levels; ++i)
                {
                    cv::cuda::GpuMat next;
                    cv::cuda::resize(current, next, cv::Size(), scale, scale, cv::INTER_LINEAR);
                    pyramid.push_back(next);
                    current = next;
                }

                return pyramid;
            }

            // Batch processing utilities
            static void batchNormalize(const std::vector<cv::cuda::GpuMat> &inputs,
                                       std::vector<cv::cuda::GpuMat> &outputs,
                                       float mean = 128.0f, float stddev = 64.0f)
            {
                outputs.resize(inputs.size());

                // Process in parallel using streams
                std::vector<cv::cuda::Stream> streams(inputs.size());

                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    cv::cuda::normalize(inputs[i], outputs[i], mean, stddev,
                                        cv::NORM_MINMAX, -1, cv::noArray(), streams[i]);
                }

                // Wait for all streams
                for (auto &stream : streams)
                {
                    stream.waitForCompletion();
                }
            }

            // Memory pool management
            static void optimizeMemoryPool()
            {
                cv::cuda::setBufferPoolUsage(true);
                cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(),
                                              1024 * 1024 * 256, // 256MB
                                              2);                // 2 stacks
            }
        };

        // Utility functions for format conversion
        __global__ void rgb2grayKernel(const uchar3 *input, uchar *output,
                                       int width, int height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width || y >= height)
                return;

            int idx = y * width + x;
            uchar3 rgb = input[idx];

            // Standard grayscale conversion: 0.299*R + 0.587*G + 0.114*B
            output[idx] = static_cast<uchar>(
                0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z);
        }

        // Fast color space conversion
        cv::cuda::GpuMat fastRGB2Gray(const cv::cuda::GpuMat &input)
        {
            CV_Assert(input.type() == CV_8UC3);

            cv::cuda::GpuMat output(input.size(), CV_8UC1);

            dim3 block(16, 16);
            dim3 grid((input.cols + block.x - 1) / block.x,
                      (input.rows + block.y - 1) / block.y);

            rgb2grayKernel<<<grid, block>>>(
                input.ptr<uchar3>(), output.ptr<uchar>(),
                input.cols, input.rows);

            cudaDeviceSynchronize();
            return output;
        }

    } // namespace cuda
} // namespace hybrid_sfm