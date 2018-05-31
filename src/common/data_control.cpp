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
#include "common/data_control.h"

#include <numeric>
#include <cassert>
#include <fstream>
#include <iomanip>

#include "common/config.h"
#include "common/constants.h"
#include "common/error.h"
#include "common/sizes.h"

#include "utilities/misc/string_utils.h"
#include "utilities/comms/config.h"

#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
#include "utilities/debug/zlib_compressor.h"
#include "utilities/debug/dump_utils.h"
#endif

#include "packages/hydro/config.h"
#include "packages/ale/config.h"



namespace bookleaf {

DataControl::DataControl() : data((int) DataID::COUNT)
{
}



#if defined BOOKLEAF_DEBUG
void
DataControl::dump(std::string filename) const
{
#ifdef BOOKLEAF_ZLIB_SUPPORT
    Error err;

#ifdef BOOKLEAF_MPI_SUPPORT
    // Serialise the dumps per-process if we're running under MPI
    int nproc, rank;
    TYPH_Get_Size(&nproc);
    TYPH_Get_Rank(&rank);

    filename += "." + std::to_string(rank) + ".bldump";

    TYPH_Barrier();
    for (int ip = 0; ip < nproc; ip++) {
        if (ip == rank) {
#endif // BOOKLEAF_MPI_SUPPORT

            std::ofstream of(filename.c_str(), std::ios_base::binary);
            if (!of.is_open()) {
                FAIL_WITH_LINE(err, "ERROR: couldn't open file " + filename);
                std::exit(EXIT_FAILURE);
            }

            ZLibCompressor zlc(of);
            zlc.init();

            // Write number of views
            SizeType const num_views = data.size();
            zlc.write((unsigned char const *) &num_views, sizeof(SizeType));

            for (SizeType i = 0; i < num_views; i++) {
                Data const &d = data[i];

                std::string const name = d.getName();
                size_type const size = d.size();

                if (d.getAllocatedType() == "integer") {
                    writeZLibDump(zlc, name, d.chost<int, VarDim>(size));

                } else if (d.getAllocatedType() == "double") {
                    writeZLibDump(zlc, name, d.chost<double, VarDim>(size));

                } else if (d.getAllocatedType() == "boolean") {
                    writeZLibDump(zlc, name, d.chost<unsigned char, VarDim>(size));
                }
            }

            zlc.finish();
            of.close();
            if (err.failed()) {
                std::exit(EXIT_FAILURE);
            }

#ifdef BOOKLEAF_MPI_SUPPORT
        }
        TYPH_Barrier();
    }
#endif // BOOKLEAF_MPI_SUPPORT
#endif // BOOKLEAF_ZLIB_SUPPORT
}
#endif // defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT



void
DataControl::setMesh(Sizes const &sizes)
{
    using bookleaf::constants::NCORN;

    // Copy local sizes
    size_type const nel = sizes.nel2;
    size_type const nnd = sizes.nnd2;

    assert(nel > 0);
    assert(nnd > 0);

    // Set mesh memory
    entry<double>(DataID::NDX, "ndx", nnd);
    entry<double>(DataID::NDY, "ndy", nnd);
    entry<int>(DataID::INDTYPE, "ndtype", nnd);
    entry<int>(DataID::IELND, "elnd", nel, NCORN);
    entry<int>(DataID::IELMAT, "elmaterial", nel);
    entry<int>(DataID::IELREG, "elregion", nel);
}



void
DataControl::setQuant(Config const &config, Sizes const &sizes)
{
    using constants::NCORN;
    using constants::NFACE;

    // Copy to local storage
    size_type  const nel  = sizes.nel2;
    size_type  const nnd  = sizes.nnd2;
    size_type  const nsz  = std::max(nel, nnd);
    bool const zmpi = config.comms->zmpi;

    assert(nel > 0);
    assert(nnd > 0);
    assert(nsz > 0);

    // Set quant. memory
    entry<double>(DataID::ELDENSITY, "eldensity", zmpi, nel);
    entry<double>(DataID::ELENERGY, "elenergy", zmpi, nel);
    entry<double>(DataID::ELPRESSURE, "elpressure", nel);
    entry<double>(DataID::ELCS2, "elcs2", zmpi, nel);
    entry<double>(DataID::ELVOLUME, "elvolume", zmpi, nel);
    entry<double>(DataID::A1, "a1", nel);
    entry<double>(DataID::A2, "a2", nel);
    entry<double>(DataID::A3, "a3", nel);
    entry<double>(DataID::B1, "b1", nel);
    entry<double>(DataID::B2, "b2", nel);
    entry<double>(DataID::B3, "b3", nel);
    entry<double>(DataID::CNWT, "cnwt", zmpi, 0, nel, NCORN);
    entry<double>(DataID::CNX, "cnx", nel, NCORN);
    entry<double>(DataID::CNY, "cny", nel, NCORN);
    entry<double>(DataID::NDU, "ndu", nnd);
    entry<double>(DataID::NDV, "ndv", nnd);
    entry<double>(DataID::RSCRATCH21, "rscratch21", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH22, "rscratch22", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH23, "rscratch23", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH24, "rscratch24", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH25, "rscratch25", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH26, "rscratch26", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH27, "rscratch27", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::RSCRATCH28, "rscratch28", zmpi, 0, nsz, NCORN);
    entry<double>(DataID::CNVISCX, "cnviscx", nel, NCORN);
    entry<double>(DataID::CNVISCY, "cnviscy", nel, NCORN);
    entry<double>(DataID::ELVISC, "elvisc", nel);
    entry<double>(DataID::ELMASS, "elmass", nel);
    entry<double>(DataID::CNMASS, "cnmass", zmpi, 0, nel, NCORN);
    entry<double>(DataID::RSCRATCH11, "rscratch11", zmpi, nsz);
    entry<double>(DataID::RSCRATCH12, "rscratch12", zmpi, nsz);
    entry<double>(DataID::RSCRATCH13, "rscratch13", zmpi, nsz);
    entry<double>(DataID::RSCRATCH14, "rscratch14", zmpi, nsz);
    entry<double>(DataID::RSCRATCH15, "rscratch15", zmpi, nsz);
    entry<double>(DataID::RSCRATCH16, "rscratch16", zmpi, nsz);
    entry<double>(DataID::RSCRATCH17, "rscratch17", zmpi, nsz);

    entry<int>(DataID::IELSORT1, "ielsort1", nel);
    entry<int>(DataID::IELEL, "ielel", nel, NFACE);
    entry<int>(DataID::IELFC, "ielfc", nel, NFACE);
    entry<int>(DataID::ISCRATCH11, "iscratch11", nsz);

    if (config.hydro->zsp) {
        entry<double>(DataID::SPMASS, "spmass", nel, NCORN);
    }

    if (config.ale->zexist) {
        entry<int>(DataID::IELSORT2, "ielsort2", nel);
        entry<unsigned char>(DataID::ZSCRATCH11, "zscratch11", nsz);
    }
}



#ifdef BOOKLEAF_MPI_SUPPORT
void
DataControl::setTyphon(Sizes const &sizes)
{
    size_type const nel2 = sizes.nel2;
    size_type const nnd2 = sizes.nnd2;

    // Set Typhon memory
    entry<int>(DataID::IELLOCGLOB, "iellocglob", nel2);
    entry<int>(DataID::INDLOCGLOB, "indlocglob", nnd2);
}
#endif



void
DataControl::resetCpQuant(size_type nsize, Sizes &sizes, Error &err)
{
    using constants::NCORN;

    // Return if new data size smaller than current data size
    if (nsize <= (size_type) sizes.mcp) return;

    // Component data
    if (sizes.mcp == 0) {
        entry<double>(DataID::CPDENSITY, "cpdensity", nsize);
        entry<double>(DataID::CPENERGY, "cpenergy", nsize);
        entry<double>(DataID::CPPRESSURE, "cppressure", nsize);
        entry<double>(DataID::CPCS2, "cpcs2", nsize);
        entry<double>(DataID::CPVOLUME, "cpvolume", nsize);
        entry<double>(DataID::FRVOLUME, "frvolume", nsize);
        entry<double>(DataID::CPMASS, "cpmass", nsize);
        entry<double>(DataID::FRMASS, "frmass", nsize);
        entry<double>(DataID::CPVISCX, "cpviscx", nsize);
        entry<double>(DataID::CPVISCY, "cpviscy", nsize);
        entry<double>(DataID::CPVISC, "cpvisc", nsize);
        entry<double>(DataID::CPA1, "cpa1", nsize);
        entry<double>(DataID::CPA3, "cpa3", nsize);
        entry<double>(DataID::CPB1, "cpb1", nsize);
        entry<double>(DataID::CPB3, "cpb3", nsize);
        entry<int>(DataID::ICPMAT, "icpmat", nsize);
        entry<int>(DataID::ICPNEXT, "icpnext", nsize);
        entry<int>(DataID::ICPPREV, "icpprev", nsize);
        entry<int>(DataID::ICPSCRATCH11, "icpscratch11", nsize);
        entry<int>(DataID::ICPSCRATCH12, "icpscratch12", nsize);
        entry<double>(DataID::RCPSCRATCH11, "rcpscratch11", nsize);
        entry<double>(DataID::RCPSCRATCH21, "rcpscratch21", nsize, NCORN);
        entry<double>(DataID::RCPSCRATCH22, "rcpscratch22", nsize, NCORN);
        entry<double>(DataID::RCPSCRATCH23, "rcpscratch23", nsize, NCORN);
        entry<double>(DataID::RCPSCRATCH24, "rcpscratch24", nsize, NCORN);

    } else {
        reset<double>(DataID::CPDENSITY, nsize);
        reset<double>(DataID::CPENERGY, nsize);
        reset<double>(DataID::CPPRESSURE, nsize);
        reset<double>(DataID::CPCS2, nsize);
        reset<double>(DataID::CPVOLUME, nsize);
        reset<double>(DataID::FRVOLUME, nsize);
        reset<double>(DataID::CPMASS, nsize);
        reset<double>(DataID::FRMASS, nsize);
        reset<double>(DataID::CPVISCX, nsize);
        reset<double>(DataID::CPVISCY, nsize);
        reset<double>(DataID::CPVISC, nsize);
        reset<double>(DataID::CPA1, nsize);
        reset<double>(DataID::CPA3, nsize);
        reset<double>(DataID::CPB1, nsize);
        reset<double>(DataID::CPB3, nsize);
        reset<int>(DataID::ICPMAT, nsize);
        reset<int>(DataID::ICPNEXT, nsize);
        reset<int>(DataID::ICPPREV, nsize);
        reset<int>(DataID::ICPSCRATCH11, nsize);
        reset<int>(DataID::ICPSCRATCH12, nsize);
        reset<double>(DataID::RCPSCRATCH11, nsize);
        reset<double>(DataID::RCPSCRATCH21, nsize, NCORN);
        reset<double>(DataID::RCPSCRATCH22, nsize, NCORN);
        reset<double>(DataID::RCPSCRATCH23, nsize, NCORN);
        reset<double>(DataID::RCPSCRATCH24, nsize, NCORN);
    }

    // Error handle
    if (err.failed()) {
        err.fail("ERROR: failed in resetCpQuant");
        return;
    }

    // Store new size
    sizes.mcp = nsize;
}



void
DataControl::resetMxQuant(size_type nsize, Sizes &sizes, Error &err)
{
    // Return if new data size smaller than current data size
    if (nsize <= (size_type) sizes.mmx) return;

    // Multi-material permanent data
    if (sizes.mmx == 0) {
        entry<int>(DataID::IMXEL, "imxel", nsize);
        entry<int>(DataID::IMXNCP, "imxncp", nsize);
        entry<int>(DataID::IMXFCP, "imxfcp", nsize);

    } else {
        reset<int>(DataID::IMXEL, nsize);
        reset<int>(DataID::IMXNCP, nsize);
        reset<int>(DataID::IMXFCP, nsize);
    }

    // Error handle
    if (err.failed()) {
        err.fail("ERROR: failed in resetMxQuant");
        return;
    }

    // Store new size
    sizes.mmx = nsize;
}



void
DataControl::syncAllDevice() const
{
    for (Data const &d : data) {
        if (d.isAllocated()) d.syncDevice();
    }
}



void
DataControl::syncAllHost() const
{
    for (Data const &d : data) {
        if (d.isAllocated()) d.syncHost();
    }
}

} // namespace bookleaf
