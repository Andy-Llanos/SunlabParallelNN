// kittiLoader.hpp
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "utils.hpp"  // for Point3D

// maxPoints == 0 -> load all; otherwise stop once we hit maxPoints
inline std::vector<Point3D> loadKittiBin(
    const std::string& filename,
    size_t maxPoints = 0)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "ERROR: cannot open KITTI file: " << filename << "\n";
        return {};
    }

    std::vector<Point3D> points;
    points.reserve(100000); // pre-reserve a bit; optional

    float data[4];
    while (in.read(reinterpret_cast<char*>(data), sizeof(data))) {
        Point3D p{data[0], data[1], data[2]};
        points.push_back(p);
        if (maxPoints > 0 && points.size() >= maxPoints)
            break;
    }

    return points;
}

////simple down sampling helper if 
inline std::vector<Point3D> downsample_stride(
    const std::vector<Point3D>& in,
    size_t maxPoints)
{
    if (maxPoints == 0 || in.size() <= maxPoints) return in;

    size_t stride = in.size() / maxPoints;
    if (stride == 0) stride = 1;

    std::vector<Point3D> out;
    out.reserve(maxPoints);
    for (size_t i = 0; i < in.size() && out.size() < maxPoints; i += stride) {
        out.push_back(in[i]);
    }
    return out;
}

