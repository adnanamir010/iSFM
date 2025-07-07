#pragma once

#include <string>
#include <vector>
#include "features.h"

namespace hybrid_sfm {

class Image {
public:
    Image() = default;

    ImageID id = 0;
    std::string path;

    // Storing observations of features detected in this image
    std::vector<Observation> observations;

    // Dimensions
    int width = 0;
    int height = 0;
};

} // namespace hybrid_sfm