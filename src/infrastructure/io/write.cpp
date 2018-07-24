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
#include "infrastructure/io/write.h"

#include <iostream>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <regex>
#include <cassert>

#ifndef BOOKLEAF_DARWIN_BUILD
#include <sched.h>  // sched_getcpu()
#include <unistd.h> // gethostname
#include <limits.h> // HOST_NAME_MAX
#endif

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include <omp.h>

#include "common/runtime.h"
#include "common/config.h"
#include "common/error.h"
#include "common/cmd_args.h"
#include "packages/time/config.h"
#include "packages/hydro/config.h"
#include "packages/hydro/driver/print.h"
#include "packages/ale/config.h"
#include "utilities/eos/config.h"
#include "utilities/data/global_configuration.h"
#include "utilities/comms/config.h"
#include "utilities/io/config.h"
#include "packages/setup/config.h"
#include "packages/setup/types.h"
#include "common/timer_control.h"
#include "infrastructure/io/output_formatting.h"

#ifdef BOOKLEAF_SILO_SUPPORT
#include "packages/io/driver/silo_io_driver.h"
#endif
#ifdef BOOKLEAF_TYPHONIO_SUPPORT
#include "packages/io/driver/typhon_io_driver.h"
#endif



namespace bookleaf {
namespace inf {
namespace io {
namespace {

void
printShortConfiguration(
        Config const &config,
        Runtime const &runtime,
        DataControl &data)
{
    // Hydro tables
    hydro::driver::shortPrint(*config.hydro, runtime, data);
}



void
printPreprocessingOptions(
#ifdef BOOKLEAF_MPI_SUPPORT
        comms::Comms const &comms)
#else
        comms::Comms const &comms __attribute__((unused)))
#endif
{
#ifdef BOOKLEAF_MPI_SUPPORT
    std::cout << "  MPI parallelism included\n";
    std::cout << "  Running " << comms.world->nproc << " MPI ranks\n";

    #ifdef BOOKLEAF_PARMETIS_SUPPORT
    std::cout << "  Metis used for decomposition\n";
    #else
    std::cout << "  Metis not used for decomposition\n";
    #endif
#else
    std::cout << "  MPI parallelism not included\n";
#endif

    std::cout << "  OpenMP variant\n";
    std::cout << "  Running " << comms.nthread << " thread(s)\n";

#ifdef BOOKLEAF_SILO_SUPPORT
    std::cout << "  Silo visualisation dumps available\n";
#else
    std::cout << "  No Silo visualisation dumps available\n";
#endif

    std::cout << inf::io::stripe() << "\n";
}



#ifndef BOOKLEAF_DARWIN_BUILD
void
printBinding(
        comms::Comm const &comm)
{
    // Get host name
    std::string hostname;
    {
        char _hostname[HOST_NAME_MAX];
        gethostname(_hostname, HOST_NAME_MAX);
        hostname = std::string(_hostname);
    }

    int nt;
    #pragma omp parallel
    {
        #pragma omp master
        nt = omp_get_num_threads();
    }

    int *thread_bind = new int[nt];

#ifdef BOOKLEAF_MPI_SUPPORT
    TYPH_Barrier();
#endif

    // Print binding
    for (int iproc = 0; iproc < comm.nproc; iproc++) {
        if (iproc == comm.rank) {

            #pragma omp parallel
            {
                thread_bind[omp_get_thread_num()] = sched_getcpu();
            }

            std::cout << "  rank " << iproc << " -> " << hostname << ":[";
            for (int it = 0; it < nt; it++) {
                std::cout << thread_bind[it] << ", ";
            }
            std::cout << "]\n";
        }

#ifdef BOOKLEAF_MPI_SUPPORT
        TYPH_Barrier();
#endif
    }

    delete[] thread_bind;

    if (comm.zmproc) {
        std::cout << inf::io::stripe() << "\n";
    }
}
#endif // !BOOKLEAF_DARWIN_BUILD



void
printFileContents(
        comms::Comm const &comm,
        std::string filename)
{
    if (!comm.zmproc) return;

    std::ifstream ifs(filename.c_str());
    assert(ifs.is_open());

    std::string contents(
            (std::istreambuf_iterator<char>(ifs)),
            std::istreambuf_iterator<char>());

    // Indent by two spaces
    contents.insert(0, "  ");
    contents = std::regex_replace(contents, std::regex("\\n"), "\n  ");

    std::cout << contents << "\n";
    std::cout << stripe() << "\n";

    ifs.close();
}

} // namespace

void
printConfiguration(
        std::string filename,
        Config const &config,
        Runtime const &runtime)
{
    if (CMD_ARGS.print_bindings) {
        if (config.comms->world->zmproc) {
            std::cout << " CPU BINDING\n";
        }

#ifndef BOOKLEAF_DARWIN_BUILD
        printBinding(*config.comms->world);
#else
        if (config.comms->world->zmproc) {
            std::cout << "  (not supported on Darwin)\n";
            std::cout << inf::io::stripe() << "\n";
        }
#endif
    }

    if (config.comms->world->zmproc) {
        {
            auto t = std::time(nullptr);
            auto tmm = *std::localtime(&t);
            std::stringstream ss;
            ss << std::put_time(&tmm, "%d/%m/%Y at %H:%M:%S");

            std::cout << inf::io::format_value("Input file", "", filename);
            std::cout << inf::io::format_value("Time stamp", "", ss.str());
            std::cout << inf::io::stripe() << "\n";
        }

        std::cout << " PRE-PROCESSING OPTIONS\n";
        printPreprocessingOptions(*config.comms);

        printFileContents(*config.comms->world, filename);

        std::cout << " TIME OPTIONS\n";
        std::cout << *config.time;

        std::cout << stripe() << "\n";
        std::cout << " HYDRO OPTIONS\n";
        std::cout << *config.hydro;

        std::cout << stripe() << "\n";
        std::cout << " ALE OPTIONS\n";
        std::cout << *config.ale;

        std::cout << stripe() << "\n";
        std::cout << " EOS OPTIONS\n";
        std::cout << *config.eos;

        std::cout << stripe() << "\n";
        std::cout << " GLOBAL OPTIONS\n";
        std::cout << *config.global;

        std::cout << stripe() << "\n";
        std::cout << " MESHING OPTIONS\n";
        printMeshRegions(config.setup->mesh_regions, *runtime.sizes);

        std::cout << stripe() << "\n";
        std::cout << " INITIAL CONDITIONS\n";
        std::cout << config.setup->regions;
        std::cout << config.setup->materials;
        std::cout << "\n";
        std::cout << config.setup->thermo;
        std::cout << config.setup->kinematics;
        std::cout << "\n";
        std::cout << config.setup->shapes;
        std::cout << stripe() << "\n";
    }
}



void
writeOutput(
        std::string dumpname,
        Config const &config,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data)
{
    ScopedTimer st(timers, TimerID::IO);
    printShortConfiguration(config, runtime, data);

    // XXX(timrlaw): otherwise gcc moans about unused argument if neither SILO
    //               nor TyphonIO are included
    std::string const dn = dumpname;

#ifdef BOOKLEAF_SILO_SUPPORT
    {
        Error err;
        bookleaf::io::SiloIODriver sio;
        sio.dump(dn, *config.io, *runtime.sizes, data, *config.comms->world,
                timers, err);
        if (err.failed()) {
            halt(config, runtime, timers, data, err);
        }
    }
#endif // BOOKLEAF_SILO_SUPPORT

#ifdef BOOKLEAF_TYPHONIO_SUPPORT
    {
        static_assert(false, "not yet implemented");
    }
#endif // BOOKLEAF_TYPHONIO_SUPPORT
}



void
printTimers(
        comms::Comms const &comms,
        TimerControl &timers)
{
    if (comms.world->zmproc) {
        std::cout << inf::io::stripe() << "\n";
        std::cout << timers;
    }
}

} // namespace io
} // namespace inf
} // namespace bookleaf
