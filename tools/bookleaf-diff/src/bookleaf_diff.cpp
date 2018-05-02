#include <iostream>
#include <string>
#include <fstream>
#include <unistd.h> // getopt

#include "data_dump.h"



void
usage(int argc, char *argv[])
{
    std::cerr << "Usage: " << argv[0] << " <1.bldump> <2.bldump>\n";
    std::cerr << "\n";
}



int
main(int argc, char *argv[])
{
    using namespace bookleaf_diff;

    // Secret options for comparing against Fortran version.
    bool ignore_off_by_one = false;
    bool multimaterial = false;

    double epsilon = 0.;

    int opt;
    while ((opt = getopt(argc, argv, "ime:")) != -1) {
        switch (opt) {
            case 'i':
                ignore_off_by_one = true;
                break;

            case 'm':
                multimaterial = true;
                break;

            case 'e':
                epsilon = std::stod(optarg);
                break;

            default:
                usage(argc, argv);
                return EXIT_FAILURE;
        }
    }

    // Parse the 2 non-optional arguments
    if (argc - optind != 2) {
        usage(argc, argv);
        return EXIT_FAILURE;
    }

    std::string filename1(argv[optind]);
    std::string filename2(argv[optind+1]);

    // Check that files are valid
    std::ifstream if1(filename1.c_str());
    if (!if1.is_open()) {
        std::cerr << "Couldn't open " << filename1 << "\n";
        return EXIT_FAILURE;
    }
    if1.close();

    std::ifstream if2(filename2.c_str());
    if (!if2.is_open()) {
        std::cerr << "Couldn't open " << filename2 << "\n";
        return EXIT_FAILURE;
    }
    if2.close();

    DiffOpts diff_opts;
    diff_opts[DiffOpt::MULTIMATERIAL].flag = multimaterial;
    diff_opts[DiffOpt::IGNORE_OFF_BY_ONE].flag = ignore_off_by_one;
    diff_opts[DiffOpt::EPSILON].rval = epsilon;

    DataDump data1;
    DataDump data2;

    // Read files
    data1.read(filename1);
    data2.read(filename2);

    // Compare the two
    bool result = data1.diff(data2, diff_opts);

    return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
