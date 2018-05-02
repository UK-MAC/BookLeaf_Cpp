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

#include "common/sizes.h"
#include "common/data_control.h"
#include "common/timer_control.h"
#include "utilities/eos/config.h"



namespace bookleaf {
namespace eos {
namespace kernel {
namespace {

inline double
getPressure(int mat, double density, double energy, EOS const &eos)
{
    double pressure = 0.;
    MaterialEOS::Type type = MaterialEOS::Type::VOID;

    int const imat = std::max(mat, 0);
    assert(imat < (int) eos.mat_eos.size());
    if (mat >= 0) {
        type = eos.mat_eos[imat].type;
    }

    double const *params = eos.mat_eos[imat].params;
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
            pressure = params[0] * (::pow(t1, params[1]) - 1.);
            pressure = std::max(pressure, params[3]);
        }
        break;

    // Jones–Wilkins–Lee
    case MaterialEOS::Type::JWL:
        {
            double t1 = params[3] * params[5] / density;
            double t2 = params[4] * params[5] / density;
            double t3 = params[0] * density * energy;
            double t4 = (1. - params[0] / t1) * params[1] * std::exp(-t1);
            double t5 = (1. - params[0] / t2) * params[2] * std::exp(-t2);
            pressure = t3 + t4 + t5;
        }
        break;

    default:
        pressure = -1.;
    }

    if (std::abs(pressure) < eos.pcut) pressure = 0.;
    return pressure;
}



inline double
getCS2(int mat, double density, double energy, EOS const &eos)
{
    double cs2 = 0.;
    MaterialEOS::Type type = MaterialEOS::Type::VOID;

    int const imat = mat;
    assert(imat < (int) eos.mat_eos.size());
    if (mat >= 0) {
        type = eos.mat_eos[imat].type;
    }

    double const *params = eos.mat_eos[imat].params;
    switch (type) {
    case MaterialEOS::Type::VOID:
        cs2 = eos.ccut;
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
            cs2 = cs2 * ::pow(t1, t2);
        }
        break;

    // Jones–Wilkins–Lee
    case MaterialEOS::Type::JWL:
        {
            double t1 = params[5] / density;
            double t2 = getPressure(imat, density, energy, eos);
            double t3 = params[3] * t1;

            double t4 = params[0] / params[3] + params[0]*t1 - t3*t1;
            t4 *= params[1] * std::exp(-t3);

            t3 = params[4] * t1;
            double t5 = params[0] / params[4] + params[0]*t1 - t3*t1;
            t5 *= params[2] * std::exp(-t3);

            cs2 = params[0] *  t2 / density + params[0] * energy - t4 - t5;
        }
        break;

    default:
        cs2 = eos.ccut;
    }

    cs2 = std::max(cs2, eos.ccut);
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

    // Update pressure and sound speed
    for (int i = 0; i < len; i++) {
        pressure(i) = getPressure(
                mat(i),
                density(i),
                energy(i),
                eos);

        cs2(i) = getCS2(
                mat(i),
                density(i),
                energy(i),
                eos);
    }
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

    auto elmat      = data[DataID::IELMAT].chost<int, VarDim>();
    auto eldensity  = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto elenergy   = data[DataID::ELENERGY].chost<double, VarDim>();
    auto elpressure = data[DataID::ELPRESSURE].host<double, VarDim>();
    auto elcs2      = data[DataID::ELCS2].host<double, VarDim>();

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
        auto cpmat      = data[DataID::ICPMAT].chost<int, VarDim>();
        auto cpdensity  = data[DataID::CPDENSITY].chost<double, VarDim>();
        auto cpenergy   = data[DataID::CPENERGY].chost<double, VarDim>();
        auto cppressure = data[DataID::CPPRESSURE].host<double, VarDim>();
        auto cpcs2      = data[DataID::CPCS2].host<double, VarDim>();

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
