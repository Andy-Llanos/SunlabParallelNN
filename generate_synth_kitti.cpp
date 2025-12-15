// generate_synth_kitti.cpp
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>   // std::snprintf

/// Compile:
///   g++ -std=c++17 -O3 generate_synth_kitti.cpp -o gen_kitti
///
/// Single frame (same as before):
///   ./gen_kitti data/kitti/velodyne/000000.bin 100000 123
///
/// Sequence mode:
///   ./gen_kitti --seq data/kitti/velodyne 50 100000 123  0.05 0.00 0.00  0.02  0.01 0.01
///                 ^     out_dir         T   N     seed   dx   dy   dz   noise  drop add
///
/// Meaning:
///   T frames, each frame applies translation (dx,dy,dz) per frame,
///   adds Gaussian noise (stddev=noise),
///   drops drop% points and adds add% new points.

struct Point3DI {
    float x, y, z, intensity;
};

static bool writeBin(const std::string& path, const std::vector<Point3DI>& pts) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(pts.data()),
              (std::streamsize)(pts.size() * sizeof(Point3DI)));
    return true;
}

static std::vector<Point3DI> makeBaseCloud(size_t N, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist_i(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist_angle(0.0f, 6.2831853f);
    std::uniform_real_distribution<float> dist_radius(5.0f, 50.0f);
    std::uniform_real_distribution<float> dist_jitter(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dist_z(-3.0f, 3.0f);

    std::vector<Point3DI> pts;
    pts.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        float angle = dist_angle(rng);
        float radius = dist_radius(rng);

        Point3DI p;
        p.x = radius * std::cos(angle) + dist_jitter(rng);
        p.y = radius * std::sin(angle) + dist_jitter(rng);
        p.z = dist_z(rng);
        p.intensity = dist_i(rng);
        pts.push_back(p);
    }
    return pts;
}

static void applyFrameUpdate(std::vector<Point3DI>& pts,
                             std::mt19937& rng,
                             float dx, float dy, float dz,
                             float noiseStd,
                             float dropFrac,
                             float addFrac)
{
    // 1) translate + gaussian noise
    std::normal_distribution<float> noise(0.0f, noiseStd);

    for (auto& p : pts) {
        p.x += dx + noise(rng);
        p.y += dy + noise(rng);
        p.z += dz + noise(rng);
        // intensity can also drift a little if you want; keep stable:
        // p.intensity = std::min(1.0f, std::max(0.0f, p.intensity + 0.01f*noise(rng)));
    }

    // 2) drop a fraction of points (simulate occlusion / fewer returns)
    if (dropFrac > 0.0f) {
        size_t dropN = (size_t)std::llround(dropFrac * (double)pts.size());
        if (dropN > 0 && dropN < pts.size()) {
            // remove by swap-with-end to avoid expensive erase in loop
            for (size_t i = 0; i < dropN; ++i) {
                std::uniform_int_distribution<size_t> pick(0, pts.size() - 1);
                size_t idx = pick(rng);
                pts[idx] = pts.back();
                pts.pop_back();
            }
        }
    }

    // 3) add a fraction of new points (simulate new returns)
    if (addFrac > 0.0f) {
        size_t addN = (size_t)std::llround(addFrac * (double)pts.size());
        if (addN > 0) {
            auto added = makeBaseCloud(addN, rng);
            pts.insert(pts.end(), added.begin(), added.end());
        }
    }
}

int main(int argc, char** argv) {
    // Sequence mode
    if (argc >= 2 && std::string(argv[1]) == "--seq") {
        if (argc < 12) {
            std::cerr << "Usage:\n  " << argv[0]
                      << " --seq <out_dir> <num_frames> <num_points> <seed>"
                      << " <dx> <dy> <dz> <noise_std> <drop_frac> <add_frac>\n";
            return 1;
        }

        std::string outDir = argv[2];
        int T = std::stoi(argv[3]);
        size_t N = std::stoul(argv[4]);
        unsigned seed = (unsigned)std::stoul(argv[5]);

        float dx = std::stof(argv[6]);
        float dy = std::stof(argv[7]);
        float dz = std::stof(argv[8]);
        float noiseStd = std::stof(argv[9]);
        float dropFrac = std::stof(argv[10]);
        float addFrac  = std::stof(argv[11]);

        std::mt19937 rng(seed);

        // Base cloud at frame 0
        std::vector<Point3DI> pts = makeBaseCloud(N, rng);

        for (int t = 0; t < T; ++t) {
            char name[64];
            std::snprintf(name, sizeof(name), "%06d.bin", t);
            std::string outPath = outDir + "/" + name;

            if (!writeBin(outPath, pts)) {
                std::cerr << "ERROR: cannot open " << outPath << " for writing\n";
                return 1;
            }

            std::cout << "Wrote frame " << t << " (" << pts.size() << " pts) to " << outPath << "\n";

            // update pts for next frame
            applyFrameUpdate(pts, rng, dx, dy, dz, noiseStd, dropFrac, addFrac);
        }

        return 0;
    }

    // Single-frame mode (backwards compatible with your original)
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <output.bin> <num_points> [seed]\n";
        std::cerr << "   or: " << argv[0]
                  << " --seq <out_dir> <num_frames> <num_points> <seed> <dx> <dy> <dz> <noise_std> <drop_frac> <add_frac>\n";
        return 1;
    }

    std::string outPath = argv[1];
    size_t numPoints = std::stoul(argv[2]);
    unsigned seed = (argc > 3) ? (unsigned)std::stoul(argv[3]) : 123u;

    std::mt19937 rng(seed);

    auto pts = makeBaseCloud(numPoints, rng);
    if (!writeBin(outPath, pts)) {
        std::cerr << "ERROR: cannot open " << outPath << " for writing\n";
        return 1;
    }

    std::cout << "Wrote " << pts.size() << " points to " << outPath << "\n";
    return 0;
}
