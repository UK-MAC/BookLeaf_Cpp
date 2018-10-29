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
#ifndef BOOKLEAF_UTILITIES_GEOMETRY_CONFIG_H
#define BOOKLEAF_UTILITIES_GEOMETRY_CONFIG_H

#include "common/defs.h"



namespace bookleaf {

struct Sizes;
struct Error;

namespace geometry {

struct Config
{
    unsigned char *cub_storage = nullptr;
    SizeType cub_storage_len = 0;
    int *cub_out = nullptr;
};

void
initGeometryConfig(
        Sizes const &sizes,
        geometry::Config &geom,
        Error &err);

void
killGeometryConfig(
        geometry::Config &geom);

} // namespace geometry
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_GEOMETRY_CONFIG_H
