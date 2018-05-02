#include "data_dump.h"

#include <cstring>
#include <cmath>
#include <memory>
#include <cassert>
#include <iostream>
#include <algorithm>

#include "trim.h"
#include "zlib_decompressor.h"



namespace bookleaf_diff {
namespace {

template <typename T>
T
get_value(void *ptr, int offset)
{
    return ((T *) ptr)[offset];
}

} // namespace

DataDump::Array const &
DataDump::get(std::string name) const
{
    for (auto const &array : arrays) {
        if (array.name == name) return array;
    }

    throw std::runtime_error(name + " array not found");
}



DataDump::Array &
DataDump::get(std::string name)
{
    for (auto &array : arrays) {
        if (array.name == name) return array;
    }

    throw std::runtime_error(name + " array not found");
}



void
DataDump::read(std::string filename)
{
    typedef std::size_t size_type;

#if 0
    std::ifstream is(filename.c_str(), std::ios_base::binary);

    size_type num_arrays;
    is.read((char *) &num_arrays, sizeof(size_type));

    std::cout << "num arrays = " << num_arrays << "\n";

    for (size_type i = 0; i < num_arrays; i++) {
        Array arr;

        size_type name_len;
        is.read((char *) &name_len, sizeof(size_type));

        std::unique_ptr<char[]> _name(new char[name_len]);
        is.read((char *) _name.get(), name_len);
        arr.name = std::string(_name.get(), name_len);

        std::cout << arr.name << "\n";

        is.read((char *) &arr.size, sizeof(size_type));

        if (arr.size == 0) continue;

        size_type type_name_len;
        is.read((char *) &type_name_len, sizeof(size_type));

        std::unique_ptr<char[]> _type_name(new char[type_name_len]);
        is.read((char *) _type_name.get(), type_name_len);
        arr.type = std::string(_type_name.get(), type_name_len);

        // Allocate space for values and read them in
        if (arr.type == "integer") {
            arr.data = malloc(sizeof(int) * arr.size);
            is.read((char *) arr.data, sizeof(int) * arr.size);

        } else if (arr.type == "double") {
            arr.data = malloc(sizeof(double) * arr.size);
            is.read((char *) arr.data, sizeof(double) * arr.size);

        } else if (arr.type == "boolean") {
            arr.data = malloc(sizeof(unsigned char) * arr.size);
            is.read((char *) arr.data, sizeof(unsigned char) * arr.size);

        } else {
            std::cerr << "Unrecognised array type '" << arr.type << "'\n";
            std::exit(EXIT_FAILURE);
        }

        for (int i = 0; i < 40; i++) {
            std::cout << ((double *) arr.data)[i] << "\n";
        }
        std::cout << "\n";

        arrays.push_back(arr);
    }

    is.close();
#else
    std::ifstream ifs(filename.c_str(), std::ios_base::binary);
    ZLibDecompressor zld(ifs);
    zld.init();

    size_type num_arrays;
    if (!zld.read((unsigned char *) &num_arrays, sizeof(size_type))) {
        std::cerr << "error\n";
    }

    for (size_type i = 0; i < num_arrays; i++) {
        Array arr;

        size_type name_len;
        zld.read((unsigned char *) &name_len, sizeof(size_type));

        std::unique_ptr<char[]> _name(new char[name_len]);
        zld.read((unsigned char *) _name.get(), name_len);
        arr.name = std::string(_name.get(), name_len);

        zld.read((unsigned char *) &arr.size, sizeof(size_type));

        if (arr.size == 0) continue;

        size_type type_name_len;
        zld.read((unsigned char *) &type_name_len, sizeof(size_type));

        std::unique_ptr<char[]> _type_name(new char[type_name_len]);
        zld.read((unsigned char *) _type_name.get(), type_name_len);
        arr.type = std::string(_type_name.get(), type_name_len);

        // Allocate space for values and read them in
        if (arr.type == "integer") {
            arr.data = malloc(sizeof(int) * arr.size);
            zld.read((unsigned char *) arr.data, sizeof(int) * arr.size);

        } else if (arr.type == "double") {
            arr.data = malloc(sizeof(double) * arr.size);
            zld.read((unsigned char *) arr.data, sizeof(double) * arr.size);

        } else if (arr.type == "boolean") {
            arr.data = malloc(sizeof(unsigned char) * arr.size);
            zld.read((unsigned char *) arr.data, sizeof(unsigned char) * arr.size);

        } else {
            std::cerr << "Unrecognised array type '" << arr.type << "'\n";
            std::exit(EXIT_FAILURE);
        }

        arrays.push_back(arr);
    }

    zld.finish();
    ifs.close();
#endif

    // Sort arrays alphabetically
    auto alphabetical_comparison = [](Array const &arr1, Array const &arr2) {
        return arr1.name.compare(arr2.name) < 0;
    };

    std::sort(arrays.begin(), arrays.end(), alphabetical_comparison);
}



bool
DataDump::diff(DataDump const &rhs, DiffOpts const &diff_opts)
{
    auto const &data1 = arrays;
    auto const &data2 = rhs.arrays;

    bool ignore_off_by_one = false;
    if (diff_opts.find(DiffOpt::IGNORE_OFF_BY_ONE) != diff_opts.end()) {
        ignore_off_by_one = diff_opts.find(DiffOpt::IGNORE_OFF_BY_ONE)->second.flag;
    }

    bool multimaterial = false;
    if (diff_opts.find(DiffOpt::MULTIMATERIAL) != diff_opts.end()) {
        multimaterial = diff_opts.find(DiffOpt::MULTIMATERIAL)->second.flag;
    }

    double epsilon = 1.e-15;
    if (diff_opts.find(DiffOpt::EPSILON) != diff_opts.end()) {
        epsilon = diff_opts.find(DiffOpt::EPSILON)->second.rval;
    }

    if (data1.size() != data2.size()) {
        std::cerr << "Different number of arrays: " << data1.size() <<
            " vs. " << data2.size() << "\n";
        return false;
    }

    for (int i = 0; i < data1.size(); i++) {
        Array const &arr1 = data1[i];
        Array const &arr2 = data2[i];

        if (arr1.name != arr2.name) {
            std::cerr << "Mismatching array name: '" << arr1.name << "' vs. '"
                << arr2.name << "'\n";
            return false;
        }

        if (arr1.size != arr2.size) {
            std::cerr << "'" << arr1.name << "' size mismatch: " << arr1.size
                << " vs. " << arr2.size << "'\n";
            return false;
        }

        if (arr1.type != arr2.type) {
            std::cerr << "'" << arr1.name << "' type mismatch: " << arr1.type
                << " vs. " << arr2.type << "'\n";
            return false;
        }

        int offset = 0;
        bool offset_set = false;
        for (int j = 0; j < arr1.size; j++) {
            bool result = false;
            std::string sv1;
            std::string sv2;

            if (arr1.type == "integer") {
                int v1 = get_value<int>(arr1.data, j);
                int v2 = get_value<int>(arr2.data, j);
                if (v1 == Array::IINIT) {
                    result = v1 == v2;

                } else if (multimaterial && arr1.name == "elmaterial" && v1 < 0) {
                    result = v1 == v2;

                } else if (multimaterial && arr1.name == "icpprev" && v1 < 0) {
                    result = v1 == v2;

                } else {
                    if (ignore_off_by_one) {
                        int const difference = v1 - v2;
                        if (!offset_set) {
                            offset = difference;
                            result = std::abs(difference) <= 1;
                            offset_set = true;
                        } else {
                            result = difference == offset;
                        }

                    } else {
                        result = v1 == v2;
                    }
                }

                sv1 = std::to_string(v1);
                sv2 = std::to_string(v2);

            } else if (arr1.type == "double") {
                double v1 = get_value<double>(arr1.data, j);
                double v2 = get_value<double>(arr2.data, j);
                result = std::fabs(v1 - v2) <= epsilon;
                sv1 = std::to_string(v1);
                sv2 = std::to_string(v2);

            } else if (arr1.type == "boolean") {
                unsigned char v1 = get_value<unsigned char>(arr1.data, j);
                unsigned char v2 = get_value<unsigned char>(arr2.data, j);
                result = v1 == v2;
                sv1 = std::to_string(v1);
                sv2 = std::to_string(v2);

            } else {
                assert(false);
            }

            if (!result) {
                std::cerr << "'" << arr1.name << "' value mismatch at " << j <<
                    ": " << sv1 << " vs. " << sv2 << "\n";
                return false;
            }
        }
    }

    std::cout << "Files match.\n";
    return true;
}

} // namespace bookleaf_diff
