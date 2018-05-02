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
#ifndef BOOKLEAF_UTILITIES_COMMS_PARTITION_H
#define BOOKLEAF_UTILITIES_COMMS_PARTITION_H

#ifndef BOOKLEAF_MPI_SUPPORT
static_assert(false, "BOOKLEAF_MPI_SUPPORT required");
#endif

#include "common/view.h"



namespace bookleaf {

struct Error;

namespace comms {

struct Comm;

/** \brief Split the mesh into strips along its longest side. */
void
initialPartition(
        int const dims[2],
        Comm const &comm,
        int &side,
        int slice[2],
        int &nel);

/**
 * \brief Given a naive partitioning of the mesh into strips, and associated
 *        connectivity data, improve the partitioning.
 */
void
improvePartition(
        int const dims[2],
        int const side,
        int const slice[2],
        int const nel,
        Comm &comm,
        int const *connectivity,
        View<int, VarDim> partitioning,
        Error &err);

} // namespace comms
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_COMMS_PARTITION_H
