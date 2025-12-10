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

inline float sqDistPointAABB(const Point3D& p,
                             const Point3D& bb_min,
                             const Point3D& bb_max) {
    float dx = 0.0f;
    if      (p.x < bb_min.x) dx = bb_min.x - p.x;
    else if (p.x > bb_max.x) dx = p.x - bb_max.x;

    float dy = 0.0f;
    if      (p.y < bb_min.y) dy = bb_min.y - p.y;
    else if (p.y > bb_max.y) dy = p.y - bb_max.y;

    float dz = 0.0f;
    if      (p.z < bb_min.z) dz = bb_min.z - p.z;
    else if (p.z > bb_max.z) dz = p.z - bb_max.z;

    return dx*dx + dy*dy + dz*dz;
}

