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
#ifndef BOOKLEAF_INDICATORS_H
#define BOOKLEAF_INDICATORS_H

#include <iostream>
#include <string>
#include <vector>



namespace bookleaf {

struct Error;

namespace setup {

struct Shape;

struct Indicator
{
    enum Type : int { UNKNOWN, MATERIAL, REGION, MESH, BACKGROUND, CELL, SHAPE };
    Type type = Type::UNKNOWN;

    // Region/Material index
    int index = -1;

    // Associated parameter
    int value = -1;

    // Human readable name
    std::string name;
};

struct Region   : public Indicator {};
struct Material : public Indicator {};

std::ostream &operator<<(std::ostream &os, std::vector<Region> const &rhs);
std::ostream &operator<<(std::ostream &os, std::vector<Material> const &rhs);

void
rationaliseRegions(
        std::vector<Region>       const &regions,
        std::vector<Material>     const &materials,
        std::vector<setup::Shape> const &shapes,
        Error &err);

void
rationaliseMaterials(
        std::vector<Material>     const &materials,
        std::vector<Region>       const &regions,
        std::vector<setup::Shape> const &shapes,
        Error &err);

} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_INDICATORS_H
