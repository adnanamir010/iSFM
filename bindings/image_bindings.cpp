#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "core/types/image.h"
#include "core/utils/image_io.h"
#include <opencv2/core/cuda.hpp>

namespace py = pybind11;

namespace hybrid_sfm {

// Helper to convert cv::Mat to numpy array
py::array mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array();
    }
    
    std::vector<size_t> shape;
    shape.push_back(mat.rows);
    shape.push_back(mat.cols);
    if (mat.channels() > 1) {
        shape.push_back(mat.channels());
    }
    
    std::vector<size_t> strides;
    strides.push_back(mat.step[0]);
    strides.push_back(mat.step[1]);
    if (mat.channels() > 1) {
        strides.push_back(mat.elemSize1());
    }
    
    return py::array(py::dtype::of<uint8_t>(), shape, strides, mat.data);
}

// Helper to convert numpy array to cv::Mat
cv::Mat numpy_to_mat(py::array_t<uint8_t> array) {
    py::buffer_info buf = array.request();
    
    int rows = buf.shape[0];
    int cols = buf.shape[1];
    int channels = buf.ndim > 2 ? buf.shape[2] : 1;
    
    cv::Mat mat(rows, cols, CV_8UC(channels), buf.ptr);
    return mat.clone();  // Return a copy to avoid memory issues
}

void bind_image(py::module& m) {
    // ImageMetadata
    py::class_<ImageMetadata>(m, "ImageMetadata")
        .def(py::init<>())
        .def_readwrite("camera_make", &ImageMetadata::camera_make)
        .def_readwrite("camera_model", &ImageMetadata::camera_model)
        .def_readwrite("focal_length_mm", &ImageMetadata::focal_length_mm)
        .def_readwrite("exposure_time", &ImageMetadata::exposure_time)
        .def_readwrite("iso_speed", &ImageMetadata::iso_speed)
        .def_readwrite("aperture", &ImageMetadata::aperture)
        .def_readwrite("capture_time", &ImageMetadata::capture_time)
        .def_readwrite("has_gps", &ImageMetadata::has_gps)
        .def_readwrite("latitude", &ImageMetadata::latitude)
        .def_readwrite("longitude", &ImageMetadata::longitude)
        .def_readwrite("altitude", &ImageMetadata::altitude)
        .def("parse_from_exif", &ImageMetadata::parseFromExif);
    
    // ImagePyramid
    py::class_<ImagePyramid>(m, "ImagePyramid")
        .def(py::init<>())
        .def("build", &ImagePyramid::build,
             py::arg("base_image"), py::arg("num_levels"), py::arg("use_gpu") = true)
        .def("get_level", [](const ImagePyramid& self, int level) {
            return mat_to_numpy(self.getLevel(level));
        })
        .def("get_num_levels", &ImagePyramid::getNumLevels)
        .def("get_scale_factor", &ImagePyramid::getScaleFactor)
        .def("release_gpu", &ImagePyramid::releaseGPU)
        .def("release_cpu", &ImagePyramid::releaseCPU)
        .def_property_readonly("num_levels", &ImagePyramid::getNumLevels);
    
    // Image class
    py::class_<Image> image_class(m, "Image");
    
    // Image::LoadFlags - bind as nested enum
    py::enum_<Image::LoadFlags>(image_class, "LoadFlags", py::arithmetic())
        .value("LAZY", Image::LoadFlags::LAZY)
        .value("LOAD_GPU", Image::LoadFlags::LOAD_GPU)
        .value("BUILD_PYRAMID", Image::LoadFlags::BUILD_PYRAMID)
        .value("EXTRACT_METADATA", Image::LoadFlags::EXTRACT_METADATA);
    
    // Image class methods
    image_class
        .def(py::init<>())
        .def(py::init<const std::string&, int>(),
             py::arg("path"), py::arg("flags") = 0)
        .def(py::init([](py::array_t<uint8_t> data, const std::string& name) {
            cv::Mat mat = numpy_to_mat(data);
            return std::make_unique<Image>(mat, name);
        }), py::arg("data"), py::arg("name") = "")
        
        // Loading methods
        .def("load", &Image::load,
             py::arg("path"), py::arg("flags") = 0)
        .def("load_lazy", &Image::loadLazy)
        .def("save", &Image::save)
        
        // GPU operations
        .def("upload_to_gpu", &Image::uploadToGPU)
        .def("download_from_gpu", &Image::downloadFromGPU)
        .def("is_on_gpu", &Image::isOnGPU)
        
        // Pyramid operations
        .def("build_pyramid", &Image::buildPyramid,
             py::arg("levels"), py::arg("use_gpu") = true)
        .def("get_pyramid_level", [](const Image& self, int level) {
            return mat_to_numpy(self.getPyramidLevel(level));
        })
        
        // Image operations
        .def("get_gray", [](const Image& self, bool use_gpu) {
            return mat_to_numpy(self.getGray(use_gpu));
        }, py::arg("use_gpu") = false)
        .def("get_resized", [](const Image& self, int width, int height, bool use_gpu) {
            return mat_to_numpy(self.getResized(width, height, use_gpu));
        }, py::arg("width"), py::arg("height"), py::arg("use_gpu") = false)
        .def("get_undistorted", [](const Image& self, const Camera& camera, bool use_gpu) {
            return mat_to_numpy(self.getUndistorted(camera, use_gpu));
        }, py::arg("camera"), py::arg("use_gpu") = true)
        .def("undistort", &Image::undistort,
             py::arg("camera"), py::arg("use_gpu") = true)
        
        // Properties
        .def_property_readonly("width", &Image::getWidth)
        .def_property_readonly("height", &Image::getHeight)
        .def_property_readonly("channels", &Image::getChannels)
        .def_property_readonly("name", &Image::getName)
        .def_property_readonly("path", &Image::getPath)
        .def_property_readonly("id", &Image::getId)
        .def_property_readonly("metadata", &Image::getMetadata,
                              py::return_value_policy::reference_internal)
        .def_property_readonly("pyramid", &Image::getPyramid,
                              py::return_value_policy::reference_internal)
        .def_property_readonly("memory_usage", &Image::getMemoryUsage)
        
        // Numpy interface
        .def("to_numpy", [](const Image& self) {
            return mat_to_numpy(self.getData());
        })
        .def_static("from_numpy", [](py::array_t<uint8_t> data, const std::string& name) {
            cv::Mat mat = numpy_to_mat(data);
            return std::make_unique<Image>(mat, name);
        }, py::arg("data"), py::arg("name") = "")
        
        // Memory management
        .def("release_gpu_memory", &Image::releaseGPUMemory)
        .def_static("get_total_gpu_memory_usage", &Image::getTotalGPUMemoryUsage);
    
    // ImageFormat enum
    py::enum_<ImageFormat>(m, "ImageFormat")
        .value("UNKNOWN", ImageFormat::UNKNOWN)
        .value("JPEG", ImageFormat::JPEG)
        .value("PNG", ImageFormat::PNG)
        .value("TIFF", ImageFormat::TIFF)
        .value("BMP", ImageFormat::BMP)
        .value("WEBP", ImageFormat::WEBP)
        .value("RAW", ImageFormat::RAW)
        .value("PFM", ImageFormat::PFM)
        .value("EXR", ImageFormat::EXR);
    
    // ImageLoader::LoadOptions
    py::class_<ImageLoader::LoadOptions>(m, "LoadOptions")
        .def(py::init<>())
        .def_readwrite("flags", &ImageLoader::LoadOptions::flags)
        .def_readwrite("max_dimension", &ImageLoader::LoadOptions::max_dimension)
        .def_readwrite("num_threads", &ImageLoader::LoadOptions::num_threads)
        .def_readwrite("skip_corrupted", &ImageLoader::LoadOptions::skip_corrupted)
        .def_readwrite("progress_callback", &ImageLoader::LoadOptions::progress_callback);
    
    // Load functions with better argument handling
    m.def("load_image", [](const std::string& path, const ImageLoader::LoadOptions& options) {
        return ImageLoader::load(path, options);
    }, py::arg("path"), py::arg("options") = ImageLoader::LoadOptions());
    
    m.def("load_images_batch", [](const std::vector<std::string>& paths, 
                                 const ImageLoader::LoadOptions& options,
                                 py::kwargs kwargs) {
        // Handle kwargs for backwards compatibility
        ImageLoader::LoadOptions opts = options;
        if (kwargs.contains("skip_corrupted")) {
            opts.skip_corrupted = py::cast<bool>(kwargs["skip_corrupted"]);
        }
        return ImageLoader::loadBatch(paths, opts);
    }, py::arg("paths"), py::arg("options") = ImageLoader::LoadOptions());
    
    m.def("load_images_from_directory", [](const std::string& directory,
                                         const ImageLoader::LoadOptions& options,
                                         const std::vector<std::string>& extensions,
                                         py::kwargs kwargs) {
        // Handle kwargs for backwards compatibility
        ImageLoader::LoadOptions opts = options;
        if (kwargs.contains("max_dimension")) {
            auto max_dim = kwargs["max_dimension"];
            if (!max_dim.is_none()) {
                opts.max_dimension = py::cast<int>(max_dim);
            }
        }
        if (kwargs.contains("num_threads")) {
            opts.num_threads = py::cast<int>(kwargs["num_threads"]);
        }
        return ImageLoader::loadDirectory(directory, opts, extensions);
    }, py::arg("directory"), 
       py::arg("options") = ImageLoader::LoadOptions(),
       py::arg("extensions") = std::vector<std::string>());
    
    m.def("detect_image_format", &ImageLoader::detectFormat);
    m.def("is_image_file", &ImageLoader::isImageFile);
    m.def("get_supported_extensions", &ImageLoader::getSupportedExtensions);
    
    m.def("load_pfm", [](const std::string& path) -> py::array {
        cv::Mat mat = ImageLoader::loadPFM(path);
        if (mat.empty()) return py::array();

        // Convert float mat to numpy
        std::vector<size_t> shape = {static_cast<size_t>(mat.rows), 
                                    static_cast<size_t>(mat.cols)};
        if (mat.channels() > 1) {
            shape.push_back(mat.channels());
        }
        
    return py::array_t<float>(shape, mat.ptr<float>());
    });    
    
    m.def("save_pfm", [](const std::string& path, py::array_t<float> data) {
        py::buffer_info buf = data.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        int channels = buf.ndim > 2 ? buf.shape[2] : 1;
        
        cv::Mat mat(rows, cols, CV_32FC(channels), buf.ptr);
        return ImageLoader::savePFM(path, mat);
    });
    
    m.def("set_max_memory_usage", &ImageLoader::setMaxMemoryUsage);
    m.def("get_current_memory_usage", &ImageLoader::getCurrentMemoryUsage);
    
    // ImageSequence
    py::class_<ImageSequence>(m, "ImageSequence")
        .def(py::init<const std::string&>())
        .def("get_frame", &ImageSequence::getFrame)
        .def("get_next_frame", &ImageSequence::getNextFrame)
        .def_property_readonly("frame_count", &ImageSequence::getFrameCount)
        .def_property_readonly("current_index", &ImageSequence::getCurrentIndex)
        .def_property_readonly("fps", &ImageSequence::getFPS)
        .def("reset", &ImageSequence::reset)
        .def("has_next", &ImageSequence::hasNext)
        .def("preload_frames", &ImageSequence::preloadFrames)
        .def("clear_cache", &ImageSequence::clearCache);
    
    // Image utilities
    m.def("compute_image_stats", [](const Image& image, bool use_gpu) {
        auto stats = image_utils::computeStats(image, use_gpu);
        
        py::dict result;
        result["mean"] = py::array_t<double>({4}, stats.mean.val);
        result["stddev"] = py::array_t<double>({4}, stats.stddev.val);
        result["min"] = stats.min;
        result["max"] = stats.max;
        result["histogram"] = mat_to_numpy(stats.histogram);
        
        return result;
    }, py::arg("image"), py::arg("use_gpu") = false);
    
    m.def("normalize_image", [](py::array_t<uint8_t> image, 
                               double target_mean, double target_std) {
        cv::Mat mat = numpy_to_mat(image);
        cv::Mat normalized = image_utils::normalizeImage(mat, target_mean, target_std);
        return mat_to_numpy(normalized);
    }, py::arg("image"), py::arg("target_mean") = 128.0, py::arg("target_std") = 64.0);
    
    m.def("batch_resize", [](py::list images, int max_dimension, bool use_gpu) {
        // Note: This is a workaround - proper implementation would handle unique_ptr list
        py::print("batch_resize: Not fully implemented for Python bindings");
    }, py::arg("images"), py::arg("max_dimension"), py::arg("use_gpu") = true);

    m.def("batch_undistort", [](py::list images, const Camera& camera, bool use_gpu) {
        // Note: This is a workaround - proper implementation would handle unique_ptr list
        py::print("batch_undistort: Not fully implemented for Python bindings");
    }, py::arg("images"), py::arg("camera"), py::arg("use_gpu") = true);

    m.def("format_memory_size", &image_utils::formatMemorySize);
    m.def("optimize_memory_layout", [](py::list images) {
        // Note: This is a workaround
        py::print("optimize_memory_layout: Not fully implemented for Python bindings");
    });
        
    // CUDA utilities
    m.def("has_cuda", []() {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    });
    
    m.def("get_cuda_device_count", []() {
        return cv::cuda::getCudaEnabledDeviceCount();
    });
}

} // namespace hybrid_sfm