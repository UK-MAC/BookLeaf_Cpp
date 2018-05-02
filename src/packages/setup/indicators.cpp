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
#include "packages/setup/indicators.h"

#include <algorithm> // std::any_of, std::all_of, std::count_if

#include "common/error.h"
#include "packages/setup/types.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {
namespace setup {

std::ostream &
operator<<(
        std::ostream &os,
        std::vector<Region> const &rhs)
{
    for (int i = 0; i < (int) rhs.size(); i++) {
        Region const &region = rhs[i];
        os << "  Region: " << i << "\n";
        os << inf::io::format_value(" Region name", "", region.name);

        switch (region.type) {
        case Region::Type::MESH:
            os << inf::io::format_value(" Region type", "", "set from MESH");
            break;

        case Region::Type::BACKGROUND:
            os << inf::io::format_value(" Region type", "", "BACKGROUND");
            break;

        case Region::Type::CELL:
            os << inf::io::format_value(" Region type", "", "CELL");
            os << inf::io::format_value(" Cell index", "", region.value);
            break;

        case Region::Type::SHAPE:
            os << inf::io::format_value(" Region type", "", "SHAPE");
            os << inf::io::format_value(" Shape index", "", region.value);
            break;

        default:
            break;
        }
    }

    return os;
}



std::ostream &
operator<<(
        std::ostream &os,
        std::vector<Material> const &rhs)
{
    for (int i = 0; i < (int) rhs.size(); i++) {
        Material const &material = rhs[i];
        os << "  Material: " << i << "\n";
        os << inf::io::format_value(" Material name", "", material.name);

        switch (material.type) {
        case Material::Type::MESH:
            os << inf::io::format_value(" Material type", "", "set from MESH");
            break;

        case Material::Type::BACKGROUND:
            os << inf::io::format_value(" Material type", "", "BACKGROUND");
            break;

        case Material::Type::CELL:
            os << inf::io::format_value(" Material type", "", "CELL");
            os << inf::io::format_value(" Cell index", "", material.value);
            break;

        case Material::Type::REGION:
            os << inf::io::format_value(" Material type", "", "REGION");
            os << inf::io::format_value(" Region index", "", material.value);
            break;

        case Material::Type::SHAPE:
            os << inf::io::format_value(" Material type", "", "SHAPE");
            os << inf::io::format_value(" Shape index", "", material.value);
            break;

        default:
            break;
        }
    }

    return os;
}



void
rationaliseRegions(
        std::vector<Region> const &regions,
        std::vector<Material> const &materials,
        std::vector<setup::Shape> const &shapes,
        Error &err)
{
    int const num_regions   = regions.size();
    int const num_materials = materials.size();
    int const num_shapes    = shapes.size();

    if (num_regions < 1) {
        err.fail("ERROR: no regions defined");
        return;
    }

    // If one region type is mesh then all of them must be
    auto is_mesh = [](Region const &region) -> bool {
        return region.type == Region::Type::MESH; };

    if (std::any_of(regions.begin(), regions.end(), is_mesh)) {
        if (!std::all_of(regions.begin(), regions.end(), is_mesh)) {
            err.fail("ERROR: cannot mix region type MESH with other region types");
            return;
        }
    }

    // If any cells or shapes are used then there must be a background
    auto is_cell = [](Region const &region) -> bool {
        return region.type == Region::Type::CELL; };
    auto is_shape = [](Region const &region) -> bool {
        return region.type == Region::Type::SHAPE; };
    auto is_background = [](Region const &region) -> bool {
        return region.type == Region::Type::BACKGROUND; };

    if ((std::any_of(regions.begin(), regions.end(), is_cell) ||
         std::any_of(regions.begin(), regions.end(), is_shape)) &&
        (!std::any_of(regions.begin(), regions.end(), is_background))) {

        err.fail("ERROR: region types CELL, SHAPE require a background type");
        return;
    }

    // Must be maximum of one background region
    if (std::count_if(regions.begin(), regions.end(), is_background) > 1) {
        err.fail("ERROR: only one region of type BACKGROUND permitted");
        return;
    }

    for (auto region : regions) {
        switch (region.type) {
        case Region::Type::MESH:
        case Region::Type::BACKGROUND:
            break;

        case Region::Type::CELL:
            if (region.value < 0) {
                err.fail("ERROR: cell value for region type out of range");
                return;
            }
            break;

        case Region::Type::MATERIAL:
            if (region.value < 0 || region.value >= num_materials) {
                err.fail("ERROR: material value for region type out of range");
                return;
            }
            break;

        case Region::Type::SHAPE:
            if (region.value < 0 || region.value >= num_shapes) {
                err.fail("ERROR: shape value for region type out of range");
                return;
            }
            break;

        default:
            err.fail("ERROR: unrecognised region type");
            return;
        }
    }
}



void
rationaliseMaterials(
        std::vector<Material> const &materials,
        std::vector<Region> const &regions,
        std::vector<setup::Shape> const &shapes,
        Error &err)
{
    int const num_materials = materials.size();
    int const num_regions   = regions.size();
    int const num_shapes    = shapes.size();

    if (num_materials < 1) {
        err.fail("ERROR: no materials defined");
        return;
    }

    // If one material type is mesh then all of them must be
    auto is_mesh = [](Material const &material) -> bool {
        return material.type == Material::Type::MESH; };

    if (std::any_of(materials.begin(), materials.end(), is_mesh)) {
        if (!std::all_of(materials.begin(), materials.end(), is_mesh)) {
            err.fail("ERROR: cannot mix material type MESH with other material "
                     "types");
            return;
        }
    }

    // If any cells or shapes are used then there must be a background
    auto is_cell = [](Material const &material) -> bool {
        return material.type == Material::Type::CELL; };
    auto is_shape = [](Material const &material) -> bool {
        return material.type == Material::Type::SHAPE; };
    auto is_background = [](Material const &material) -> bool {
        return material.type == Material::Type::BACKGROUND; };

    if ((std::any_of(materials.begin(), materials.end(), is_cell) ||
         std::any_of(materials.begin(), materials.end(), is_shape)) &&
        (!std::any_of(materials.begin(), materials.end(), is_background))) {

        err.fail("ERROR: material types CELL, SHAPE require a background type");
        return;
    }

    // Must be maximum of one background region
    if (std::count_if(materials.begin(), materials.end(), is_background) > 1) {
        err.fail("ERROR: only one material of type BACKGROUND permitted");
        return;
    }

    for (auto material : materials) {
        switch (material.type) {
        case Material::Type::MESH:
        case Material::Type::BACKGROUND:
            break;

        case Material::Type::CELL:
            if (material.value < 0) {
                err.fail("ERROR: cell value for material type out of range");
                return;
            }
            break;

        case Material::Type::REGION:
            if (material.value < 0 || material.value >= num_regions) {
                err.fail("ERROR: region value for material type out of range");
                return;
            }
            break;

        case Material::Type::SHAPE:
            if (material.value < 0 || material.value >= num_shapes) {
                err.fail("ERROR: shape value for material type out of range");
                return;
            }
            break;

        default:
            err.fail("ERROR: unrecognised material type");
            return;
        }
    }
}

} // namespace setup
} // namespace bookleaf
