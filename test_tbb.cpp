#include <iostream>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

int main() {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, 5),
        [&](const tbb::blocked_range<int>& r){
            for(int i = r.begin(); i != r.end(); ++i)
                std::cout << i << "\n";
        }
    );
}
