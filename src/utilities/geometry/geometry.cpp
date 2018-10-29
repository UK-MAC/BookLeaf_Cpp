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
#include "utilities/geometry/geometry.h"

#include <cassert>
#include <iomanip>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include <cub/device/device_reduce.cuh>

#include "common/error.h"
#include "common/config.h"
#include "common/sizes.h"
#include "common/runtime.h"
#include "common/timestep.h"
#include "utilities/data/gather.h"
#include "utilities/geometry/config.h"
#include "common/data_control.h"
#include "common/timer_control.h"
#include "common/cuda_utils.h"



namespace bookleaf {
namespace geometry {
namespace driver {

void
getGeometry(
        geometry::Config const &geom,
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data,
        Error &err)
{
    using constants::NCORN;

    ScopedTimer st(timers, timerid);

    int const nel = sizes.nel;

    auto cnx     = data[DataID::CNX].cdevice<double, VarDim, NCORN>();
    auto cny     = data[DataID::CNY].cdevice<double, VarDim, NCORN>();
    auto cnwt    = data[DataID::CNWT].device<double, VarDim, NCORN>();
    auto a1      = data[DataID::A1].device<double, VarDim>();
    auto a2      = data[DataID::A2].device<double, VarDim>();
    auto a3      = data[DataID::A3].device<double, VarDim>();
    auto b1      = data[DataID::B1].device<double, VarDim>();
    auto b2      = data[DataID::B2].device<double, VarDim>();
    auto b3      = data[DataID::B3].device<double, VarDim>();
    auto volume  = data[DataID::ELVOLUME].device<double, VarDim>();
    auto scratch = data[DataID::ISCRATCH11].device<int, VarDim>();

    // Gather position to element
    utils::driver::cornerGather(sizes, DataID::NDX, DataID::CNX, data);
    utils::driver::cornerGather(sizes, DataID::NDY, DataID::CNY, data);

    // Calculate iso-parametric terms
    kernel::getIso(cnx, cny, a1, a2, a3, b1, b2, b3, cnwt, nel);

    // Calculate volume
    kernel::getVolume(a1, a3, b1, b3, volume, nel);

    if (sizes.ncp > 0) {
        /*
        auto mxel     = data[DataID::IMXEL].cdevice<int, VarDim>();
        auto mxfcp    = data[DataID::IMXFCP].cdevice<int, VarDim>();
        auto mxncp    = data[DataID::IMXNCP].cdevice<int, VarDim>();
        auto cpa1     = data[DataID::CPA1].device<double, VarDim>();
        auto cpa3     = data[DataID::CPA3].device<double, VarDim>();
        auto cpb1     = data[DataID::CPB1].device<double, VarDim>();
        auto cpb3     = data[DataID::CPB3].device<double, VarDim>();
        auto cpvolume = data[DataID::CPVOLUME].device<double, VarDim>();
        auto frvolume = data[DataID::FRVOLUME].cdevice<double, VarDim>();

        // Gather iso-parametric terms to component
        utils::kernel::mxGather<double>(sizes.nmx, mxel, mxfcp, mxncp, a1, a3,
                b1, b3, cpa1, cpa3, cpb1, cpb3);

        // Calculate component volume
        kernel::getVolume(cpa1, cpa3, cpb1, cpb3, cpvolume, sizes.ncp);

        for (int icp = 0; icp < sizes.ncp; icp++) {
            cpvolume(icp) *= frvolume(icp);
        }*/
    }

    int const vol_err = kernel::checkVolume(geom, 0.0, volume, scratch, nel);
    if (vol_err != -1) {
        FAIL_WITH_LINE(err, "ERROR: negative volume in element " +
                std::to_string(vol_err));
        return;
    }
}



void
getVertex(
        Runtime const &runtime,
        DataControl &data)
{
    double const dt = 0.5 * runtime.timestep->dt;
    int const   nnd = runtime.sizes->nnd;

    auto ndu = data[DataID::NDU].cdevice<double, VarDim>();
    auto ndv = data[DataID::NDV].cdevice<double, VarDim>();
    auto ndx = data[DataID::NDX].device<double, VarDim>();
    auto ndy = data[DataID::NDY].device<double, VarDim>();

    // Update vextex positions
    kernel::getVertex(dt, ndu, ndv, ndx, ndy, nnd);
}

} // namespace driver

namespace kernel {

void
getIso(
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        DeviceView<double, VarDim>             a1,
        DeviceView<double, VarDim>             a2,
        DeviceView<double, VarDim>             a3,
        DeviceView<double, VarDim>             b1,
        DeviceView<double, VarDim>             b2,
        DeviceView<double, VarDim>             b3,
        DeviceView<double, VarDim, NCORN>      cnwt,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    double constexpr ONE_BY_NINE = 1.0/9.0;

    dispatchCuda(
            nel,
            [=] __device__ (int const i)
    {
        a1(i) = 0.25 * (-cnx(i, 0) + cnx(i, 1) + cnx(i, 2) - cnx(i, 3));
        a2(i) = 0.25 * ( cnx(i, 0) - cnx(i, 1) + cnx(i, 2) - cnx(i, 3));
        a3(i) = 0.25 * (-cnx(i, 0) - cnx(i, 1) + cnx(i, 2) + cnx(i, 3));

        b1(i) = 0.25 * (-cny(i, 0) + cny(i, 1) + cny(i, 2) - cny(i, 3));
        b2(i) = 0.25 * ( cny(i, 0) - cny(i, 1) + cny(i, 2) - cny(i, 3));
        b3(i) = 0.25 * (-cny(i, 0) - cny(i, 1) + cny(i, 2) + cny(i, 3));

        cnwt(i, 0) = ONE_BY_NINE *
            ((3.0*b3(i) - b2(i)) * (3.0*a1(i) - a2(i))
            -(3.0*a3(i) - a2(i)) * (3.0*b1(i) - b2(i)));

        cnwt(i, 1) = ONE_BY_NINE *
            ((3.0*b3(i) + b2(i)) * (3.0*a1(i) - a2(i))
            -(3.0*a3(i) + a2(i)) * (3.0*b1(i) - b2(i)));

        cnwt(i, 2) = ONE_BY_NINE *
            ((3.0*b3(i) + b2(i)) * (3.0*a1(i) + a2(i))
            -(3.0*a3(i) + a2(i)) * (3.0*b1(i) + b2(i)));

        cnwt(i, 3) = ONE_BY_NINE *
            ((3.0*b3(i) - b2(i)) * (3.0*a1(i) + a2(i))
            -(3.0*a3(i) - a2(i)) * (3.0*b1(i) + b2(i)));
    });

    cudaSync();
}



void
getVolume(
        ConstDeviceView<double, VarDim> a1,
        ConstDeviceView<double, VarDim> a3,
        ConstDeviceView<double, VarDim> b1,
        ConstDeviceView<double, VarDim> b3,
        DeviceView<double, VarDim>      volume,
        int len)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    dispatchCuda(
            len,
            [=] __device__ (int const i)
    {
        volume(i) = 4.0 * ((a1(i) * b3(i)) - (a3(i) * b1(i)));
    });

    cudaSync();
}



int
checkVolume(
        geometry::Config const &geom,
        double val,
        ConstDeviceView<double, VarDim> volume,
        DeviceView<int, VarDim>         scratch,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    dispatchCuda(
            nel,
            [=] __device__ (int const i)
    {
        scratch(i) = volume(i) < val ? i : nel;
    });

    cudaDeviceSynchronize();

    auto &cub_storage_len = const_cast<SizeType &>(geom.cub_storage_len);
    cub::DeviceReduce::Min(
            geom.cub_storage,
            cub_storage_len,
            scratch.data(),
            geom.cub_out,
            nel);

    cudaDeviceSynchronize();

    int res;
    cudaMemcpy(&res, geom.cub_out, sizeof(int), cudaMemcpyDeviceToHost);

    return res < nel ? res : -1;
}



void
getFluxVolume(
        double cut,
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<double, VarDim>        ndx0,
        ConstDeviceView<double, VarDim>        ndy0,
        ConstDeviceView<double, VarDim>        ndx1,
        ConstDeviceView<double, VarDim>        ndy1,
        DeviceView<double, VarDim, NFACE>      fcdv,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Initialise
    dispatchCuda(
            nel,
            [=] __device__ (int const iel)
    {
        fcdv(iel, 0) = 0.;
        fcdv(iel, 1) = 0.;
        fcdv(iel, 2) = 0.;
        fcdv(iel, 3) = 0.;
    });

    // Construct volumes
    dispatchCuda(
            nel,
            [=] __device__ (int const iel)
    {
        for (int ifc = 0; ifc < NFACE; ifc++) {

            int const jp = (ifc + 1) % NCORN;
            int const n1 = elnd(iel, ifc);
            int const n2 = elnd(iel, jp);

            double const x1 = ndx0(n1);
            double const x2 = ndx0(n2);
            double const y1 = ndy0(n1);
            double const y2 = ndy0(n2);
            double const x3 = ndx1(n2);
            double const x4 = ndx1(n1);
            double const y3 = ndy1(n2);
            double const y4 = ndy1(n1);

            double const a1 = (-x1-x4)+(x3+x2);
            double const a3 = (-x1+x4)+(x3-x2);
            double const b1 = (-y1-y4)+(y3+y2);
            double const b3 = (-y1+y4)+(y3-y2);

            fcdv(iel, ifc) = 0.25 * (a1*b3 - a3*b1);
            fcdv(iel, ifc) = fcdv(iel, ifc) < cut ? 0. : fcdv(iel, ifc);
        }
    });

    cudaSync();
}



void
getVertex(
        double dt,
        ConstDeviceView<double, VarDim> ndu,
        ConstDeviceView<double, VarDim> ndv,
        DeviceView<double, VarDim>      ndx,
        DeviceView<double, VarDim>      ndy,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    dispatchCuda(
            nnd,
            [=] __device__ (int const ind)
    {
        ndx(ind) += dt * ndu(ind);
        ndy(ind) += dt * ndv(ind);
    });

    cudaSync();
}

} // namespace kernel
} // namespace geometry
} // namespace bookleaf
