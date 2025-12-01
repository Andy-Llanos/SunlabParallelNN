#pragma once
#include <cmath>

struct Point3D {
    float x, y, z;
};

inline float sqDist(const Point3D& a, const Point3D& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}
