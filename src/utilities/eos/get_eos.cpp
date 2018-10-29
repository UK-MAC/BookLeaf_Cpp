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

#include <cassert>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/sizes.h"
#include "common/data_control.h"
#include "common/timer_control.h"
#include "common/cuda_utils.h"
#include "utilities/eos/config.h"



namespace bookleaf {
namespace eos {
namespace kernel {
namespace {

inline __device__
double
getPressure(
        int imat,
        MaterialEOS::Type type,
        ConstDeviceView<double, VarDim, MaterialEOS::NUM_PARAMS> eos_params,
        double pcut,
        double density,
        double energy)
{
    double pressure = 0.;

    auto params = eos_params;

    switch (type) {
    case MaterialEOS::Type::VOID:
        pressure = 0.;
        break;

    // Ideal gas law
    case MaterialEOS::Type::IDEAL_GAS:
        // Assume a perfect gas => ideal gas law: P = rho*e(gamma-1)
        pressure = energy * density * (params(imat, 0) - 1.);
        break;

    // Tait equation (Murnaghan)
    case MaterialEOS::Type::TAIT:
        {
            double t1 = density / params(imat, 2);
            pressure = params(imat, 0) * (::pow(t1, params(imat, 1)) - 1.);
            pressure = max(pressure, params(imat, 3));
        }
        break;

    // Jones–Wilkins–Lee
    case MaterialEOS::Type::JWL:
        {
            double t1 = params(imat, 3) * params(imat, 5) / density;
            double t2 = params(imat, 4) * params(imat, 5) / density;
            double t3 = params(imat, 0) * density * energy;
            double t4 = (1. - params(imat, 0) / t1) *
                        params(imat, 1) * exp(-t1);
            double t5 = (1. - params(imat, 0) / t2) *
                        params(imat, 2) * exp(-t2);
            pressure = t3 + t4 + t5;
        }
        break;

    default:
        pressure = -1.;
    }

    if (fabs(pressure) < pcut) pressure = 0.;
    return pressure;
}



inline __device__
double
getCS2(
        int imat,
        MaterialEOS::Type type,
        ConstDeviceView<double, VarDim, MaterialEOS::NUM_PARAMS> eos_params,
        double pcut,
        double ccut,
        double density,
        double energy)
{
    double cs2 = 0.;

    auto params = eos_params;

    switch (type) {
    case MaterialEOS::Type::VOID:
        cs2 = ccut;
        break;

    // Ideal gas law
    case MaterialEOS::Type::IDEAL_GAS:
        // Speed of sound squared: gamma*(gamma-1)*e
        cs2 = params(imat, 0) * (params(imat, 0) - 1.) * energy;
        break;

    // Tait equation
    case MaterialEOS::Type::TAIT:
        {
            double t1 = density / params(imat, 2);
            double t2 = params(imat, 1) - 1.;
            cs2 = (params(imat, 0) * params(imat, 1)) / params(imat, 3);
            cs2 = cs2 * ::pow(t1, t2);
        }
        break;

    // Jones–Wilkins–Lee
    case MaterialEOS::Type::JWL:
        {
            double t1 = params(imat, 5) / density;
            double t2 = getPressure(imat, type, eos_params, pcut, density, energy);
            double t3 = params(imat, 3) * t1;

            double t4 = params(imat, 0) / params(imat, 3) +
                        params(imat, 0)*t1 - t3*t1;
            t4 *= params(imat, 1) * exp(-t3);

            t3 = params(imat, 4) * t1;
            double t5 = params(imat, 0) / params(imat, 4) +
                        params(imat, 0)*t1 - t3*t1;
            t5 *= params(imat, 2) * exp(-t3);

            cs2 = params(imat, 0) *  t2 / density + params(imat, 0) * energy -
                  t4 - t5;
        }
        break;

    default:
        cs2 = ccut;
    }

    cs2 = max(cs2, ccut);
    return cs2;
}

} // namespace

void
getEOS(
        double pcut,
        double ccut,
        ConstDeviceView<int, VarDim>                             eos_types,
        ConstDeviceView<double, VarDim, MaterialEOS::NUM_PARAMS> eos_params,
        ConstDeviceView<int, VarDim>                             mat,
        ConstDeviceView<double, VarDim>                          density,
        ConstDeviceView<double, VarDim>                          energy,
        DeviceView<double, VarDim>                               pressure,
        DeviceView<double, VarDim>                               cs2,
        int len)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Update pressure and sound speed
    dispatchCuda(
            len,
            [=] __device__ (int const i)
    {
        int const imat = mat(i);

        pressure(i) = getPressure(
                imat,
                (MaterialEOS::Type) eos_types(imat),
                eos_params,
                pcut,
                density(i),
                energy(i));

        cs2(i) = getCS2(
                imat,
                (MaterialEOS::Type) eos_types(imat),
                eos_params,
                pcut,
                ccut,
                density(i),
                energy(i));
    });

    cudaSync();
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

    ConstDeviceView<int, VarDim> eos_types(eos.types, sizes.nmat);
    ConstDeviceView<double, VarDim, MaterialEOS::NUM_PARAMS> eos_params(
            eos.params, sizes.nmat);

    auto elmat      = data[DataID::IELMAT].cdevice<int, VarDim>();
    auto eldensity  = data[DataID::ELDENSITY].cdevice<double, VarDim>();
    auto elenergy   = data[DataID::ELENERGY].cdevice<double, VarDim>();
    auto elpressure = data[DataID::ELPRESSURE].device<double, VarDim>();
    auto elcs2      = data[DataID::ELCS2].device<double, VarDim>();

    // Update pressure and sound speed
    kernel::getEOS(
            eos.pcut,
            eos.ccut,
            eos_types,
            eos_params,
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
                eos.pcut,
                eos.ccut,
                eos_types,
                eos_params,
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
