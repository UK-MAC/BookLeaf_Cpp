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
#ifndef BOOKLEAF_PACKAGES_SETUP_TYPES_H
#define BOOKLEAF_PACKAGES_SETUP_TYPES_H

#include <iostream>
#include <vector>
#include <functional>



namespace bookleaf {

struct Sizes;
struct Error;
class DataControl;
enum class DataID : int;

namespace setup {

// Class of functions that check for a point's membership within a shape, given
// some parameters.
typedef std::function<bool(
            double const *,
            double const *
        )> InsideFunc;

// Class of functions that set region/material flags based on a given shape.
// TODO(timrlaw): Change this to kernel interface, take view rather than ID
typedef std::function<void(
            int,
            double const *,
            DataID,
            Sizes &,
            DataControl &,
            InsideFunc
        )> ApplyShapeFunc;



struct Shape
{
    static constexpr int NUM_PARAMS = 4;
    enum Type : int { UNKNOWN, RECTANGLE, CIRCLE };

    Type type;
    double params[NUM_PARAMS];

    void rationalise(Error &err);
};



struct ThermodynamicsIC
{
    enum class Type        : int { UNKNOWN, REGION, MATERIAL };
    enum class EnergyScale : int { UNKNOWN, ZERO, VOLUME, MASS };

    Type                type = Type::UNKNOWN;
    int                value = -1;
    EnergyScale energy_scale = EnergyScale::UNKNOWN;
    double           density = -1.;
    double            energy = -1.;
};



struct KinematicsIC
{
    static constexpr int NUM_PARAMS = 3;

    enum class Type     : int { UNKNOWN, BACKGROUND, REGION };
    enum class Geometry : int { UNKNOWN, RADIAL, PLANAR };

    Type                 type = Type::UNKNOWN;
    Geometry         geometry = Geometry::UNKNOWN;
    int                 value = -1;
    double params[NUM_PARAMS] = {0};
};



std::ostream &
operator<<(std::ostream &os, std::vector<ThermodynamicsIC> const &rhs);

std::ostream &
operator<<(std::ostream &os, std::vector<KinematicsIC> const &rhs);

} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_TYPES_H
