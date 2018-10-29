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
#include "packages/hydro/driver/print.h"

#include <cassert>
#include <memory>

#include "common/sizes.h"
#include "common/runtime.h"
#include "common/timestep.h"
#include "packages/hydro/config.h"
#include "packages/hydro/kernel/print.h"
#include "utilities/io/config.h"
#include "utilities/comms/config.h"
#include "common/data_control.h"
#include "utilities/data/gather.h"
#include "utilities/data/global_configuration.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
shortPrint(hydro::Config const &hydro, Runtime const &runtime,
        DataControl &data)
{
    using constants::NCORN;

    auto constexpr CNU  = DataID::RSCRATCH21;
    auto constexpr CNV  = DataID::RSCRATCH22;
    auto constexpr CPWT = DataID::RCPSCRATCH21;
    auto constexpr CPU  = DataID::RCPSCRATCH22;
    auto constexpr CPV  = DataID::RCPSCRATCH23;

    auto elreg      = data[DataID::IELREG].chost<int, VarDim>();
    auto elmat      = data[DataID::IELMAT].chost<int, VarDim>();
    auto elenergy   = data[DataID::ELENERGY].chost<double, VarDim>();
    auto eldensity  = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto elmass     = data[DataID::ELMASS].chost<double, VarDim>();
    auto elvolume   = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto elpressure = data[DataID::ELPRESSURE].chost<double, VarDim>();
    auto cnwt       = data[DataID::CNWT].chost<double, VarDim, NCORN>();
    auto cnu        = data[CNU].chost<double, VarDim, NCORN>();
    auto cnv        = data[CNV].chost<double, VarDim, NCORN>();

    int const nreg = runtime.sizes->nreg;
    int const nmat = runtime.sizes->nmat;

    kernel::Flags reg(nreg);
    kernel::Flags mat(nmat);

    // Gather velocity to elements
    utils::driver::hostCornerGather(*runtime.sizes, DataID::NDU, CNU, data);
    utils::driver::hostCornerGather(*runtime.sizes, DataID::NDV, CNV, data);

    // Calculate table values
    kernel::initShortPrint(reg);
    kernel::initShortPrint(mat);

    kernel::calcShortPrint(runtime.sizes->nel, hydro.global->dencut, elreg,
            elenergy, eldensity, elmass, elvolume, elpressure, cnwt, cnu, cnv,
            reg);

    kernel::calcShortPrint(runtime.sizes->nel, hydro.global->dencut, elmat,
            elenergy, eldensity, elmass, elvolume, elpressure, cnwt, cnu, cnv,
            mat);

    if (runtime.sizes->ncp > 0) {
        auto mxel       = data[DataID::IMXEL].chost<int, VarDim>();
        auto mxfcp      = data[DataID::IMXFCP].chost<int, VarDim>();
        auto mxncp      = data[DataID::IMXNCP].chost<int, VarDim>();
        auto cpmat      = data[DataID::ICPMAT].chost<int, VarDim>();
        auto cpdensity  = data[DataID::CPDENSITY].chost<double, VarDim>();
        auto cpenergy   = data[DataID::CPENERGY].chost<double, VarDim>();
        auto cppressure = data[DataID::CPPRESSURE].chost<double, VarDim>();
        auto cpvolume   = data[DataID::CPVOLUME].chost<double, VarDim>();
        auto cpmass     = data[DataID::CPMASS].chost<double, VarDim>();
        auto cpwt       = data[CPWT].host<double, VarDim, NCORN>();
        auto cpu        = data[CPU].host<double, VarDim, NCORN>();
        auto cpv        = data[CPV].host<double, VarDim, NCORN>();

        utils::kernel::mxComponentCornerGather<double>(runtime.sizes->nmx, mxel,
                mxfcp, mxncp, cnwt, cpwt);
        utils::kernel::mxComponentCornerGather<double>(runtime.sizes->nmx, mxel,
                mxfcp, mxncp, cnu, cpu);
        utils::kernel::mxComponentCornerGather<double>(runtime.sizes->nmx, mxel,
                mxfcp, mxncp, cnv, cpv);

        auto ccpwt = data[CPWT].chost<double, VarDim, NCORN>();
        auto ccpu  = data[CPU].chost<double, VarDim, NCORN>();
        auto ccpv  = data[CPV].chost<double, VarDim, NCORN>();

        kernel::calcShortPrint(runtime.sizes->ncp, hydro.global->dencut, cpmat,
                cpenergy, cpdensity, cpmass, cpvolume, cppressure, ccpwt, ccpu,
                ccpv, mat);
    }

    // Reduction operation
#ifdef BOOKLEAF_MPI_SUPPORT
    if (hydro.comm->nproc > 1) {
        reg.reduce();
        mat.reduce();
    }
#endif

    // Calculate averages
    kernel::averageShortPrint(hydro.global->dencut, reg);
    kernel::averageShortPrint(hydro.global->dencut, mat);

    // Calculate totals
    double tot_vol, tot_mass, tot_ie, tot_ke, tot_pre, tot_density;
    if (runtime.sizes->nreg ==
            std::min(runtime.sizes->nreg, runtime.sizes->nmat)) {
        totalShortPrint(reg, tot_vol, tot_mass, tot_ie, tot_ke, tot_pre,
                tot_density);
    } else {
        totalShortPrint(mat, tot_vol, tot_mass, tot_ie, tot_ke, tot_pre,
                tot_density);
    }

    // Print values
    if (hydro.comm->zmproc) {
        kernel::writeHeaderShortPrint(runtime.timestep->nstep,
                runtime.timestep->time);

        kernel::writeTableShortPrint(
                " Table 1: Hydro region  ",
                "    region",
                hydro.io->sregions,
                reg);
        kernel::writeTableShortPrint(
                " Table 2: Hydro material  ",
                "  material",
                hydro.io->smaterials,
                mat);

        kernel::writeTotalShortPrint(tot_vol, tot_mass, tot_ie, tot_ke, tot_pre,
                tot_density);
    }
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
