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
#include "infrastructure/io/read.h"

#include <iostream>
#include <cassert>
#include <fstream>
#include <streambuf>
#include <algorithm>

#include "common/runtime.h"
#include "common/config.h"
#include "common/error.h"
#include "common/sizes.h"
#include "common/timer_control.h"

#include "utilities/io/config.h"
#include "utilities/comms/config.h"
#include "infrastructure/io/input_deck.h"
#include "infrastructure/io/output_formatting.h"

#include "packages/setup/indicators.h"
#include "packages/setup/mesh_region.h"
#include "packages/setup/config.h"

#include "packages/setup/generate_mesh.h"



namespace bookleaf {
namespace inf {
namespace io {

void
readInputDeck(
        std::string filename,
        Config &config,
        Runtime &runtime,
        Error &err)
{
    InputDeck deck;
    deck.open(filename, err);
    if (err.failed()) return;

    // Read utility configuration (setup_utils)
    deck.readGlobalConfiguration(*config.global, err);
    if (err.failed()) return;

    deck.readEOS(*config.eos, err);
    if (err.failed()) return;

    // Read general configuration (control_nml, ale_nml)
    deck.readTimeConfiguration(*config.time, err);
    if (err.failed()) return;

    deck.readHydroConfiguration(*config.hydro, err);
    if (err.failed()) return;

    deck.readALEConfiguration(*config.ale, err);
    if (err.failed()) return;

    // Read mesh descriptor
    config.setup->mesh_descriptor.reset(new setup::MeshDescriptor());
    deck.readMesh(*config.setup->mesh_descriptor);

    runtime.sizes->nel =
        config.setup->mesh_descriptor->dims[0] *
        config.setup->mesh_descriptor->dims[1];

    runtime.sizes->nnd =
        (config.setup->mesh_descriptor->dims[0]+1) *
        (config.setup->mesh_descriptor->dims[1]+1);

    if (runtime.sizes->nel == 0 || runtime.sizes->nnd == 0) {
        FAIL_WITH_LINE(err, "ERROR: zero sized mesh");
        return;
    }

    // Read region/material configuration (setup_IC_read)
    deck.readShapes(config.setup->shapes);
    deck.readIndicators(config.setup->regions, config.setup->materials);
    deck.readInitialConditions(config.setup->thermo, config.setup->kinematics);

    // Number of regions (not counting those of unknown type)
    auto const &regions = config.setup->regions;
    runtime.sizes->nreg = std::count_if(regions.begin(), regions.end(),
            [](setup::Region const &region) -> bool {
                return region.type != setup::Region::Type::UNKNOWN;
            });

    // Number of materials (not counting those of unknown type)
    auto const &materials = config.setup->materials;
    runtime.sizes->nmat = std::count_if(materials.begin(), materials.end(),
            [](setup::Material const &material) -> bool {
                return material.type != setup::Material::Type::UNKNOWN;
            });

    // Check for mesh specification
    config.setup->rmesh = std::any_of(regions.begin(), regions.end(),
            [](setup::Region const &region) -> bool {
                return region.type == setup::Region::Type::MESH;
            });

    config.setup->mmesh = std::any_of(materials.begin(), materials.end(),
            [](setup::Material const &material) -> bool {
                return material.type == setup::Material::Type::MESH;
            });

    // Set region/material labels
    config.io->sregions.clear();
    for (auto const &region : regions) {
        config.io->sregions.push_back(region.name);
    }

    config.io->smaterials.clear();
    for (auto const &material : materials) {
        config.io->smaterials.push_back(material.name);
    }
}

} // namespace io
} // namespace inf
} // namespace bookleaf
