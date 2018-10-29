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
#ifndef BOOKLEAF_UTILITIES_EOS_CONFIG_H
#define BOOKLEAF_UTILITIES_EOS_CONFIG_H

#include <vector>
#include <iostream>



namespace bookleaf {

struct Sizes;
struct Error;

/** \brief One Equation-of-State per material required. */
struct MaterialEOS
{
    enum class Type : int { UNKNOWN = -1, VOID = 0, IDEAL_GAS, TAIT, JWL };
    Type type = Type::UNKNOWN;

    static int constexpr NUM_PARAMS = 6;
    double params[NUM_PARAMS] = {0};
};



/** \brief Store EoS configuration. */
struct EOS
{
    double ccut = 1.e-6;    // Sound-speed cutoff
    double pcut = 1.e-8;    // Pressure cutoff

    std::vector<MaterialEOS> mat_eos;

    int *types = nullptr;
    double *params = nullptr;
};



std::ostream &
operator<<(
        std::ostream &os,
        MaterialEOS const &rhs);

std::ostream &
operator<<(
        std::ostream &os,
        EOS const &rhs);

void
rationalise(
        EOS &eos,
        int num_materials,
        Error &err);

void
initEOSConfig(
        Sizes const &sizes,
        EOS &eos,
        Error &err);

void
killEOSConfig(
        EOS &eos);

} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_EOS_CONFIG_H
