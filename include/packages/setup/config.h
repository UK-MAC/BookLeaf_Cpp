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
#ifndef BOOKLEAF_PACKAGES_SETUP_CONFIG_H
#define BOOKLEAF_PACKAGES_SETUP_CONFIG_H

#include <vector>
#include <memory>

#include "packages/setup/types.h"
#include "packages/setup/indicators.h"
#include "packages/setup/mesh_region.h"



namespace bookleaf {
namespace setup {

// Store data required for the setup package in a single location
struct Config {
    std::vector<Shape>            shapes;
    std::vector<Region>           regions;
    std::vector<Material>         materials;
    std::vector<ThermodynamicsIC> thermo;
    std::vector<KinematicsIC>     kinematics;

    std::unique_ptr<MeshDescriptor> mesh_descriptor;
    std::unique_ptr<MeshData>       mesh_data;

    bool rmesh = false;
    bool mmesh = false;
};



void
rationalise(setup::Config &setup, Error &err);

} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_CONFIG_H
