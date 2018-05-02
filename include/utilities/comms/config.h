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
#ifndef BOOKLEAF_UTILITIES_COMMS_CONFIG_H
#define BOOKLEAF_UTILITIES_COMMS_CONFIG_H

#include <memory>

// If MPI is not built in, transparently handle the serial case.
#ifdef BOOKLEAF_MPI_SUPPORT
#include <mpi.h>
#include "utilities/comms/exchange.h"
#endif



namespace bookleaf {

struct Error;

namespace comms {

#ifndef BOOKLEAF_MPI_SUPPORT
typedef int MPI_Comm;
#endif

/** \brief Abstraction of an MPI communicator. */
struct Comm
{
    int     nproc = 1;      //!< Number of processors in communicator
    MPI_Comm comm = 0;      //!< MPI communicator handle
    int      rank = 0;      //!< This processor's rank
    bool   zmproc = true;   //!< Is this processor master?

    int key_comm_cells = -1;        //!< Typhon key set ID

#ifdef BOOKLEAF_MPI_SUPPORT
    std::vector<CommPhase> phases;  //!< Keep track of communication phases
#endif

    // Factory methods for a communicator
    static Comm *setComm();
    static Comm *nullComm();
};



/** \brief Collection of MPI communicators. */
struct Comms
{
    std::shared_ptr<Comm> world;
    std::shared_ptr<Comm> spatial;
    std::shared_ptr<Comm> replicant;

    bool zmpi    = false;   //!< Is there more than one rank?
    int  nthread = 1;       //!< Number of threads per rank
};



/** \brief Initialise comms. */
void
initComms(
        Comms &comms,
        Error &err);

/** \brief Shutdown comms. */
void
killComms(
        Error &err);

} // namespace comms
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_COMMS_CONFIG_H
