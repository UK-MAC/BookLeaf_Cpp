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
#include "utilities/comms/config.h"

#include <numeric>
#include <iostream>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/error.h"



namespace bookleaf {
namespace comms {

Comm *
Comm::setComm()
{
    Comm *comm = new Comm();

#ifdef BOOKLEAF_MPI_SUPPORT
    TYPH_Get_Size(&comm->nproc);
    TYPH_Get_Rank(&comm->rank);
    TYPH_Set_Comm(&comm->comm);
#else
    comm->nproc = 1;
    comm->rank = 0;
    comm->comm = -1;
#endif

    comm->zmproc = comm->rank == 0;

    return comm;
}



Comm *
Comm::nullComm()
{
    Comm *comm = new Comm();

    comm->nproc = 1;
    comm->rank  = -1;
#ifdef BOOKLEAF_MPI_SUPPORT
    TYPH_Set_Comm_Self(&comm->comm);
#else
    comm->comm = -1;
#endif
    comm->zmproc = false;

    return comm;
}



void
initComms(
        Comms &comms,
#ifdef BOOKLEAF_MPI_SUPPORT
        Error &err)
#else
        Error &err __attribute__((unused)))
#endif
{
    // Initialise MPI
#ifdef BOOKLEAF_MPI_SUPPORT
    int typh_err = TYPH_Init();

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Init failed");
        err.iout = Error::ErrorHalt::HALT_ALL;
        return;
    }
#endif

    comms.world.reset(comms::Comm::setComm());
    comms.spatial.reset(comms::Comm::nullComm());
    comms.replicant.reset(comms::Comm::nullComm());

    comms.zmpi    = comms.world->nproc > 1;
    comms.nthread = 1;
}



void
killComms(
#ifdef BOOKLEAF_MPI_SUPPORT
        Error &err)
#else
        Error &err __attribute__((unused)))
#endif
{
    #ifdef BOOKLEAF_MPI_SUPPORT
    int typh_err = TYPH_Kill();

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Kill failed");
        err.iout = Error::ErrorHalt::HALT_ALL;
        return;
    }
    #endif
}

} // namespace comms
} // namespace bookleaf
