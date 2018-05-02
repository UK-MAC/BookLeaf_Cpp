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
#ifndef BOOKLEAF_COMMON_CONSTANTS_H
#define BOOKLEAF_COMMON_CONSTANTS_H

#include "common/defs.h"



namespace bookleaf {
namespace constants {

/** \brief Rank of the simulation. */
int constexpr NDIM  = 2;

/** \brief Mesh element # faces. */
int constexpr NFACE = 4;

/** \brief Mesh element # vertices. */
int constexpr NCORN = 4;

// TODO(timrlaw): Can we not remove all references to NFACE, since any polygon
// is always going to have the same number of vertices and faces?
static_assert(NFACE == NCORN, "true by definition");

/** \brief Floating point comparison epsilon. */
double constexpr EPSILON = 1e-10;

/** \brief Mesh connectivity field count. */
int constexpr NDAT = 7;

} // namespace constants
} // namespace bookleaf



#endif // BOOKLEAF_COMMON_CONSTANTS_H
