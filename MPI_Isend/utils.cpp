#include<vector>
#include <mpi.h>

/// @brief Given the vectors first, second, modifies first as [first, second]
template<typename T>
void my_append(std::vector<T>& first, const std::vector<T>& second){
    first.reserve(first.size() + second.size());
    first.insert(first.end(), second.begin(), second.end());
}

/// @brief Given the vectors first, second, modifies second as [first, second]
template<typename T>
void my_insert(const std::vector<T>& first, std::vector<T>& second){
    second.reserve(first.size() + second.size());
    second.insert(second.begin(), first.begin(), first.end());
}