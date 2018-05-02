/* @HEADER@
 * Crown Copyright 2018 AWE.
 *
 * This file is part of BookLeaf.
 *
 * BookLeaf is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * BookLeaf is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * BookLeaf. If not, see http://www.gnu.org/licenses/.
 * @HEADER@ */
#include "common/cmd_args.h"

#include <unistd.h>
#include <iostream>



namespace bookleaf {

CmdArgs *CmdArgs::singleton = nullptr;



bool
CmdArgs::init(int argc, char **argv)
{
    kill();

    // Attempt to read command-line arguments
    singleton = new CmdArgs();
    if (!singleton->read(argc, argv)) {
        delete singleton;
        singleton = nullptr;
        return false;
    }

    return true;
}



CmdArgs *
CmdArgs::get()
{
    return singleton;
}



void
CmdArgs::kill()
{
    if (singleton != nullptr) {
        delete singleton;
        singleton = nullptr;
    }
}



CmdArgs::CmdArgs()
{
}



CmdArgs::~CmdArgs()
{
}



void
CmdArgs::setDefaults()
{
    input_deck_file = "control.yaml";
    overwrite_dumps = false;
    print_bindings = false;

#ifdef BOOKLEAF_DEBUG
    timestep_cap = -1;
    dump_mesh    = false;
    dump_init    = false;
    dump_getdt   = false;
    dump_lagstep = false;
    dump_alestep = false;
    dump_final   = false;
#endif
}



bool
CmdArgs::read(int argc, char **argv)
{
    setDefaults();

#ifndef BOOKLEAF_DEBUG
    std::string opt_str("fp");
#else
    std::string opt_str("fpc:d:");
#endif

    int opt;
    while ((opt = getopt(argc, argv, opt_str.c_str())) != -1) {
        switch (opt) {
            case 'f':
                overwrite_dumps = true;
                break;

            case 'p':
                print_bindings = true;
                break;

#ifdef BOOKLEAF_DEBUG
            case 'c':
                timestep_cap = std::stoi(optarg);
                break;

            case 'd':
            {
                std::string dumps(optarg);
                while (true) {
                    std::size_t comma_pos = dumps.find_first_of(',');
                    std::string dump = dumps.substr(0, comma_pos);
                    dumps.erase(0, comma_pos + 1);

                    if      (dump == "mesh")    dump_mesh    = true;
                    else if (dump == "init")    dump_init    = true;
                    else if (dump == "getdt")   dump_getdt   = true;
                    else if (dump == "lagstep") dump_lagstep = true;
                    else if (dump == "alestep") dump_alestep = true;
                    else if (dump == "final")   dump_final   = true;
                    else {
                        std::cout << "Unrecognised dump specifier '" << dump
                                  << "'\n";
                        printUsage(argv[0]);
                        return false;
                    }

                    if (comma_pos == std::string::npos) break;
                }
                break;
            }
#endif

            default:
                printUsage(argv[0]);
                return false;
        }
    }

    // Input deck as final positional argument
    if (argc - optind == 1) {
        input_deck_file = std::string(argv[optind]);
    }

    return true;
}



void
CmdArgs::printUsage(std::string invocation)
{
    std::cerr << "Usage: " << invocation << " ";

#ifndef BOOKLEAF_DEBUG
    std::cerr << "[-fp]";
#else
    std::cerr << "[-fp] [-c <n>] [-d <dump-specifier-list>]";
#endif

    std::cerr << " [input_deck.yaml]\n";
    std::cerr << "\n";
    std::cerr << "\t-f\toverwrite existing dump files (default: off)\n";
    std::cerr << "\t-p\tprint cpu bindings at start of run (default: off)\n";
    std::cerr << "\n";

#ifdef BOOKLEAF_DEBUG
    std::cerr << "Debug options:\n";
    std::cerr << "\t-c\tstop after at most this many timesteps (default: -1)\n";
    std::cerr << "\t-d\tselect points to dump state (comma-delimited list: "
              << "any of mesh,init,getdt,lagstep,alestep,final)\n";
    std::cerr << "\n";
#endif
}

} // namespace bookleaf
