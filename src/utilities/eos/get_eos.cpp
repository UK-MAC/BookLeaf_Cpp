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
#include "utilities/eos/get_eos.h"

#include <cmath>
#include <cassert>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/cuda_utils.h"
#include "common/sizes.h"
#include "common/data_control.h"
#include "common/timer_control.h"
#include "utilities/eos/config.h"



namespace bookleaf {
namespace eos {
namespace kernel {
namespace {

BOOKLEAF_DEVICE_FUNCTION
double
getPressure(
        int const *mat_types,
        double const *mat_params,
        double pcut,
        int mat,
        double density,
        double energy)
{
    double pressure = 0.;
    MaterialEOS::Type type = MaterialEOS::Type::VOID;

    int const imat = BL_MAX(mat, 0);
    if (mat >= 0) {
        type = (MaterialEOS::Type) mat_types[imat];
    }

    double const *params = &mat_params[imat*MaterialEOS::NUM_PARAMS];
    switch (type) {
    case MaterialEOS::Type::VOID:
        pressure = 0.;
        break;

    // Ideal gas law
    case MaterialEOS::Type::IDEAL_GAS:
        // Assume a perfect gas => ideal gas law: P = rho*e(gamma-1)
        pressure = energy * density * (params[0] - 1.);
        break;

    // Tait equation (Murnaghan)
    case MaterialEOS::Type::TAIT:
        {
            double t1 = density / params[2];
            pressure = params[0] * (BL_POW(t1, params[1]) - 1.);
            pressure = BL_MAX(pressure, params[3]);
        }
        break;

    // Jones–Wilkins–Lee
    case MaterialEOS::Type::JWL:
        {
            double t1 = params[3] * params[5] / density;
            double t2 = params[4] * params[5] / density;
            double t3 = params[0] * density * energy;
            double t4 = (1. - params[0] / t1) * params[1] * BL_EXP(-t1);
            double t5 = (1. - params[0] / t2) * params[2] * BL_EXP(-t2);
            pressure = t3 + t4 + t5;
        }
        break;

    default:
        pressure = -1.;
    }

    if (BL_FABS(pressure) < pcut) pressure = 0.;
    return pressure;
}



BOOKLEAF_DEVICE_FUNCTION
double
getCS2(
        int const *mat_types,
        double const *mat_params,
        double pcut,
        double ccut,
        int mat,
        double density,
        double energy)
{
    double cs2 = 0.;
    MaterialEOS::Type type = MaterialEOS::Type::VOID;

    int const imat = mat;
    if (mat >= 0) {
        type = (MaterialEOS::Type) mat_types[imat];
    }

    double const *params = &mat_params[imat*MaterialEOS::NUM_PARAMS];
    switch (type) {
    case MaterialEOS::Type::VOID:
        cs2 = ccut;
        break;

    // Ideal gas law
    case MaterialEOS::Type::IDEAL_GAS:
        // Speed of sound squared: gamma*(gamma-1)*e
        cs2 = params[0] * (params[0] - 1.) * energy;
        break;

    // Tait equation
    case MaterialEOS::Type::TAIT:
        {
            double t1 = density / params[2];
            double t2 = params[1] - 1.;
            cs2 = (params[0] * params[1]) / params[3];
            cs2 = cs2 * BL_POW(t1, t2);
        }
        break;

    // Jones–Wilkins–Lee
    case MaterialEOS::Type::JWL:
        {
            double t1 = params[5] / density;
            double t2 = getPressure(mat_types, mat_params, pcut, imat, density, energy);
            double t3 = params[3] * t1;

            double t4 = params[0] / params[3] + params[0]*t1 - t3*t1;
            t4 *= params[1] * BL_EXP(-t3);

            t3 = params[4] * t1;
            double t5 = params[0] / params[4] + params[0]*t1 - t3*t1;
            t5 *= params[2] * BL_EXP(-t3);

            cs2 = params[0] *  t2 / density + params[0] * energy - t4 - t5;
        }
        break;

    default:
        cs2 = ccut;
    }

    cs2 = BL_MAX(cs2, ccut);
    return cs2;
}

} // namespace

void
getEOS(
        EOS const &eos,
        ConstView<int, VarDim>    mat,
        ConstView<double, VarDim> density,
        ConstView<double, VarDim> energy,
        View<double, VarDim>      pressure,
        View<double, VarDim>      cs2,
        int len)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    int const *mat_types = eos.d_mat_types;
    double const *mat_params = eos.d_mat_params;
    double const pcut = eos.pcut;
    double const ccut = eos.ccut;

    // Update pressure and sound speed
    RAJA::forall<RAJA_POLICY>(
            RAJA::RangeSegment(0, len),
            BOOKLEAF_DEVICE_LAMBDA (int const i)
    {
        pressure(i) = getPressure(
                mat_types,
                mat_params,
                pcut,
                mat(i),
                density(i),
                energy(i));

        cs2(i) = getCS2(
                mat_types,
                mat_params,
                pcut,
                ccut,
                mat(i),
                density(i),
                energy(i));
    });
}

} // namespace kernel

namespace driver {

void
getEOS(
        EOS const &eos,
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data)
{
    ScopedTimer st(timers, timerid);

    int const nel = sizes.nel;

    // XXX Missing code here that can't be merged

    auto elmat      = data[DataID::IELMAT].cdevice<int, VarDim>();
    auto eldensity  = data[DataID::ELDENSITY].cdevice<double, VarDim>();
    auto elenergy   = data[DataID::ELENERGY].cdevice<double, VarDim>();
    auto elpressure = data[DataID::ELPRESSURE].device<double, VarDim>();
    auto elcs2      = data[DataID::ELCS2].device<double, VarDim>();

    // Update pressure and sound speed
    kernel::getEOS(
            eos,
            elmat,
            eldensity,
            elenergy,
            elpressure,
            elcs2,
            nel);

    if (sizes.ncp > 0) {
        auto cpmat      = data[DataID::ICPMAT].cdevice<int, VarDim>();
        auto cpdensity  = data[DataID::CPDENSITY].cdevice<double, VarDim>();
        auto cpenergy   = data[DataID::CPENERGY].cdevice<double, VarDim>();
        auto cppressure = data[DataID::CPPRESSURE].device<double, VarDim>();
        auto cpcs2      = data[DataID::CPCS2].device<double, VarDim>();

        kernel::getEOS(
                eos,
                cpmat,
                cpdensity,
                cpenergy,
                cppressure,
                cpcs2,
                sizes.ncp);
    }

    // XXX Missing code here that can't be merged
}

} // namespace driver
} // namespace eos
} // namespace bookleaf
