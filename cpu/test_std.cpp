#include <string>
#include <random>
#include <iostream>

int main() {
    std::string s = "hi";
    std::random_device rd;
    std::cout << s << " " << rd() << "\n";
    return 0;
}
