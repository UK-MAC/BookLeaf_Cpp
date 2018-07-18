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
#include "packages/setup/types.h"

#include "common/error.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {
namespace setup {

void
Shape::rationalise(Error &err) const
{
    switch (type) {
    case Shape::Type::RECTANGLE:
        break;

    case Shape::Type::CIRCLE:
        if (params[2] < 0.) {
            err.fail("ERROR: shape radius < 0");
            return;
        }
        break;

    default:
        err.fail("ERROR: unrecognised shape type");
        return;
    }

    // Check fixed volume fraction
    if (!(0.0 <= params[4] && params[4] <= 1.0)) {
        err.fail("ERROR: shape fixed fraction <0 or >1");
        return;
    }
}



std::ostream &
operator<<(std::ostream &os, std::vector<Shape> const &rhs)
{
    for (int i = 0; i < (int) rhs.size(); i++) {
        Shape const &shape = rhs[i];
        os << "  Shape: " << i << "\n";

        std::string pos;
        switch (shape.type) {
            case Shape::Type::RECTANGLE:
                os << inf::io::format_value(" Shape type", "", "RECTANGLE");
                pos = "(" + std::to_string(shape.params[0]) + ", " +
                            std::to_string(shape.params[1]) + ", " +
                            std::to_string(shape.params[2]) + ", " +
                            std::to_string(shape.params[3]) + ")";
                os << inf::io::format_value(" Shape position", "", pos);
                break;

            case Shape::Type::CIRCLE:
                os << inf::io::format_value(" Shape type", "", "CIRCLE");
                pos = "(" + std::to_string(shape.params[0]) + ", " +
                            std::to_string(shape.params[1]) + ", " +
                            std::to_string(shape.params[2]) + ")";
                os << inf::io::format_value(" Shape position", "", pos);
                break;

            default:
                os << inf::io::format_value(" Shape type", "", "UNKNOWN");
                break;
        }

        if (shape.params[4] > 0.0) {
            os << inf::io::format_value(" Fixed user defined volume fraction",
                    "", shape.params[4]);
        }
    }

    return os;
}



std::ostream &
operator<<(std::ostream &os, std::vector<ThermodynamicsIC> const &rhs)
{
    for (int i = 0; i < (int) rhs.size(); i++) {
        ThermodynamicsIC const &tic = rhs[i];
        os << "  Thermodynamic IC: " << i << "\n";

        switch (tic.type) {
        case ThermodynamicsIC::Type::REGION:
            os << inf::io::format_value(" Region index", "value", tic.value);
            os << inf::io::format_value(" Density", "density", tic.density);
            os << inf::io::format_value(" Energy", "energy", tic.energy);
            break;

        case ThermodynamicsIC::Type::MATERIAL:
            os << inf::io::format_value(" Material index", "value", tic.value);
            os << inf::io::format_value(" Density", "density", tic.density);
            os << inf::io::format_value(" Energy", "energy", tic.energy);
            break;

        default:
            break;
        }

        switch (tic.energy_scale) {
        case ThermodynamicsIC::EnergyScale::VOLUME:
            os << inf::io::format_value("", "", "Energy scaled by volume");
            break;

        case ThermodynamicsIC::EnergyScale::MASS:
            os << inf::io::format_value("", "", "Energy scaled by mass");
            break;

        default:
            break;
        }
    }

    return os;
}



std::ostream &
operator<<(std::ostream &os, std::vector<KinematicsIC> const &rhs)
{
    for (int i = 0; i < (int) rhs.size(); i++) {
        KinematicsIC const &kic = rhs[i];
        os << "  Kinematic IC: " << i << "\n";

        switch (kic.type) {
        case KinematicsIC::Type::BACKGROUND:
            os << inf::io::format_value(" Background", "", "");
            break;

        case KinematicsIC::Type::REGION:
            os << inf::io::format_value(" Region index", "value", kic.value);
            break;

        default:
            break;
        }

        switch (kic.geometry) {
        case KinematicsIC::Geometry::RADIAL:
            os << inf::io::format_value(" Radial velocity profile", "velocity",
                    kic.params[0]);
            os << inf::io::format_value("", "centre",
                    "(" + std::to_string(kic.params[1]) + ", " +
                          std::to_string(kic.params[2]) + ")");
            break;

        case KinematicsIC::Geometry::PLANAR:
            os << inf::io::format_value(" Planar velocity profile", "velocity",
                    "(" + std::to_string(kic.params[0]) + ", " +
                          std::to_string(kic.params[1]) + ")");
            break;

        default:
            break;
        }
    }

    return os;
}


} // namespace setup
} // namespace bookleaf
