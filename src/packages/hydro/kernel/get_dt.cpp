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
#include "packages/hydro/kernel/get_dt.h"

#include <limits>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include <cub/device/device_reduce.cuh>

#include "common/constants.h"
#include "common/error.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"
#include "packages/hydro/config.h"



namespace bookleaf {
namespace hydro {
namespace kernel {
namespace {

inline __device__ double
denom(double x1, double y1, double x2, double y2)
{
    double w1 = y1 - y2;
    double w2 = x1 - x2;
    return w1*w1 + w2*w2;
}



inline __device__ double
distpp(double x3, double y3, double x4, double y4, double x1, double y1)
{
    double w1 = 0.5 * (x3 + x4) - x1;
    double w2 = 0.5 * (y3 + y4) - y1;
    return w1*w1 + w2*w2;
}



inline __device__ double
distpl(double x3, double y3, double x4, double y4, double x1, double y1,
        double x2, double y2)
{
    double w1 = 0.5 * (y1 - y2) * (x3 + x4) +
                0.5 * (y3 + y4) * (x2 - x1) + y2 * x1 - y1*x2;
    return w1*w1;
}



// These two kernels were originally located in the geometry utility
inline __device__ void
dlm(
        int iel,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        double res[NCORN])
{
    double elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    double x1 = elx[0] + elx[1];
    double x2 = elx[2] + elx[3];
    double y1 = ely[0] + ely[1];
    double y2 = ely[2] + ely[3];

    x1 = 0.5 * (x1 - x2);
    y1 = 0.5 * (y1 - y2);
    res[0] = x1*x1 + y1*y1;

    x1 = elx[2] + elx[1];
    x2 = elx[0] + elx[3];
    y1 = ely[2] + ely[1];
    y2 = ely[0] + ely[3];

    x1 = 0.5 * (x1 - x2);
    y1 = 0.5 * (y1 - y2);

    res[1] = x1*x1 + y1*y1;
    res[2] = res[0];
    res[3] = res[1];
}



inline __device__ void
dln(
        double zcut,
        int iel,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        double res[NCORN])
{
    double elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    for (int i = 0; i < NCORN; i++) {
        int const j1 = i;
        int const j2 = (i + 1) % NCORN;
        int const j3 = (i + 2) % NCORN;
        int const j4 = (i + 3) % NCORN;

        double const w1 = denom(elx[j3], ely[j3], elx[j4], ely[j4]);

        double const w2 = w1 < zcut ?
            distpp(elx[j1], ely[j1], elx[j2],
                   ely[j2], elx[j3], ely[j3]) :
            distpl(elx[j1], ely[j1], elx[j2],
                   ely[j2], elx[j3], ely[j3],
                   elx[j4], ely[j4]) / w1;

        res[i] = w2;
    }
}

} // namespace

void
getDtCfl(
        hydro::Config const &hydro,
        int nel,
        double zcut,
        double cfl_sf,
        ConstDeviceView<unsigned char, VarDim> zdtnotreg,
        ConstDeviceView<unsigned char, VarDim> zmidlength,
        ConstDeviceView<int, VarDim>           elreg,
        ConstDeviceView<double, VarDim>        elcs2,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        DeviceView<double, VarDim>             rscratch11,
        DeviceView<double, VarDim>             rscratch12,
        double &rdt,
        int &idt,
        std::string &sdt,
        Error &err)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Calculate CFL condition
    dispatchCuda(
            nel,
            [=] __device__ (int const iel)
    {
        int ireg = elreg(iel);

        double resultm[NCORN];
        dlm(iel, cnx, cny, resultm);

        double resultn[NCORN];
        dln(zcut, iel, cnx, cny, resultn);

        // Minimise result
        double *result = zmidlength(ireg) ? resultm : resultn;
        double w1 = result[0];
        for (int i = 1; i < NCORN; i++) {
            w1 = (result[i] < w1) ? result[i] : w1;
        }

        rscratch11(iel) = zdtnotreg(ireg) ?
                            NPP_MAXABS_64F : w1/elcs2(iel);
        rscratch12(iel) = zdtnotreg(ireg) ?
                            NPP_MINABS_64F : w1;
    });

    cudaDeviceSynchronize();

    // Find minimum CFL condition
    auto &cub_storage_len = const_cast<SizeType &>(hydro.cub_storage_len);
    auto cuda_err = cub::DeviceReduce::ArgMin(
            hydro.cub_storage,
            cub_storage_len,
            rscratch11.data(),
            hydro.cub_out,
            nel);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: Reduction failed");
        return;
    }

    cudaDeviceSynchronize();

    cub::KeyValuePair<int, double> res;
    cudaMemcpy(&res, hydro.cub_out, sizeof(cub::KeyValuePair<int, double>),
            cudaMemcpyDeviceToHost);

    double w1 = res.value;
    if (w1 < 0) {
        FAIL_WITH_LINE(err, "ERROR");
        return;
    }

    rdt = cfl_sf*sqrt(w1);
    idt = res.key;
    sdt = "     CFL";
}



void
getDtDiv(
        hydro::Config const &hydro,
        int nel,
        double div_sf,
        ConstDeviceView<double, VarDim>        a1,
        ConstDeviceView<double, VarDim>        a3,
        ConstDeviceView<double, VarDim>        b1,
        ConstDeviceView<double, VarDim>        b3,
        ConstDeviceView<double, VarDim>        elvolume,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim>             scratch,
        double &rdt,
        int &idt,
        std::string &sdt)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    dispatchCuda(
            nel,
            [=] __device__ (int const iel)
    {
        double w1 = cnu(iel, 0) * (-b3(iel) + b1(iel)) +
                    cnv(iel, 0) * ( a3(iel) - a1(iel)) +
                    cnu(iel, 1) * ( b3(iel) + b1(iel)) +
                    cnv(iel, 1) * (-a3(iel) - a1(iel)) +
                    cnu(iel, 2) * ( b3(iel) - b1(iel)) +
                    cnv(iel, 2) * (-a3(iel) + a1(iel)) +
                    cnu(iel, 3) * (-b3(iel) - b1(iel)) +
                    cnv(iel, 3) * ( a3(iel) + a1(iel));

        scratch(iel) = fabs(w1) / elvolume(iel);
    });

    cudaDeviceSynchronize();

    // Reduce scratch
    auto &cub_storage_len = const_cast<SizeType &>(hydro.cub_storage_len);
    cub::DeviceReduce::ArgMax(
            hydro.cub_storage,
            cub_storage_len,
            scratch.data(),
            hydro.cub_out,
            nel);

    cudaDeviceSynchronize();

    cub::KeyValuePair<int, double> res;
    cudaMemcpy(&res, hydro.cub_out, sizeof(cub::KeyValuePair<int, double>),
            cudaMemcpyDeviceToHost);

    rdt = div_sf/res.value;
    idt = res.key;
    sdt = "     DIV";
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
