// generate_synth_kitti.cpp
#include <iostream>
#include <fstream>
#include <random>
#include <string>
///g++ -std=c++17 -O3 generate_synth_kitti.cpp -o gen_kitti
//./gen_kitti data/kitti/velodyne/000000.bin 100000
//./gen_kitti data/kitti/velodyne/000001.bin 80000  124   # different size + seed
//./gen_kitti data/kitti/velodyne/000xxx.bin #numPts SEED#

// Simple synthetic point
struct Point3DI {
    float x, y, z, intensity;
};
//
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <output.bin> <num_points> [seed]\n";
        return 1;
    }

    std::string outPath = argv[1];
    size_t numPoints = std::stoul(argv[2]);
    unsigned seed = (argc > 3) ? static_cast<unsigned>(std::stoul(argv[3])) : 123u;

    std::mt19937 rng(seed);

    // You can tweak these distributions to look more "LiDAR-ish"
    std::uniform_real_distribution<float> dist_xy(-50.0f, 50.0f);  // +/- 50m
    std::uniform_real_distribution<float> dist_z(-3.0f, 3.0f);     // ground-ish
    std::uniform_real_distribution<float> dist_i(0.0f, 1.0f);      // intensity

    std::ofstream out(outPath, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: cannot open " << outPath << " for writing\n";
        return 1;
    }

    for (size_t i = 0; i < numPoints; ++i) {
        Point3DI p;
        // Make a rough "ring" structure around the origin
        float angle = std::uniform_real_distribution<float>(0.0f, 6.2831853f)(rng);
        float radius = std::uniform_real_distribution<float>(5.0f, 50.0f)(rng);

        p.x = radius * std::cos(angle) + std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
        p.y = radius * std::sin(angle) + std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
        p.z = dist_z(rng);
        p.intensity = dist_i(rng);

        out.write(reinterpret_cast<const char*>(&p), sizeof(Point3DI));
    }

    std::cout << "Wrote " << numPoints << " points to " << outPath << "\n";
    return 0;
}
