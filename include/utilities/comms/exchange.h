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
#ifndef BOOKLEAF_UTILITIES_COMMS_EXCHANGE_H
#define BOOKLEAF_UTILITIES_COMMS_EXCHANGE_H

#ifndef BOOKLEAF_MPI_SUPPORT
static_assert(false, "BOOKLEAF_MPI_SUPPORT required");
#endif

#include <vector>

#include "common/defs.h"
#include "common/data_id.h"
#include "common/timer_control.h"



namespace bookleaf {

struct Error;
class DataControl;

namespace comms {

struct Comm;

/** \brief Identify instances of CommPhase. */
typedef SizeType CommPhaseID;

/** \brief Store information relevant to individual Typhon phases. */
struct CommPhase
{
    int typh_id = -1;               //!< Typhon handle for phase
    std::vector<DataID> data_ids;   //!< Data communicated by this phase
};



/** \brief Mark the start of the comms initialisation. */
void
startCommsInit(
        Error &err);

/** \brief Mark the end of the comms initialisation. */
void
finishCommsInit(
        Error &err);

/** \brief Create a new comms phase. */
CommPhaseID
addCommPhase(
        Comm &comm,
        std::string name,
        int num_ghost_layers,
        Error &err);

/** \brief Add a data item to an existing comms phase. */
void
addDataToCommPhase(
        Comm &comm,
        CommPhaseID comm_phase_id,
        DataID data_id,
        DataControl const &data,
        Error &err);

/** \brief Execute a comms phase. */
void
exchange(
        Comm const &comm,
        CommPhaseID comm_phase_id,
        TimerID timer_id,
        TimerControl &timers,
        DataControl const &data,
        Error &err);

} // namespace comms
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_COMMS_EXCHANGE_H
