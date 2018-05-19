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
#include "utilities/comms/exchange.h"

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/error.h"
#include "common/data_control.h"

#include "utilities/comms/config.h"



namespace bookleaf {
namespace comms {
namespace {

bool
inRegistration()
{
    int registration;
    TYPH_Is_Registering(&registration);
    return registration == 1;
}

} // namespace

void
startCommsInit(
        Error &err)
{
    if (inRegistration()) {
        FAIL_WITH_LINE(err, "ERROR: Typhon already in registration mode");
        return;
    }

    // Start Typhon registry
    int typh_err = TYPH_Start_Register();

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Start_Register failed");
        return;
    }
}



void
finishCommsInit(
        Error &err)
{
    if (!inRegistration()) {
        FAIL_WITH_LINE(err, "ERROR: Typhon not in registration mode");
        return;
    }

    // Finalise Typhon registry
    int typh_err = TYPH_Finish_Register();

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Finish_Register failed");
        return;
    }
}



CommPhaseID
addCommPhase(
        Comm &comm,
        std::string name,
        int num_ghost_layers,
        Error &err)
{
    if (!inRegistration()) {
        FAIL_WITH_LINE(err, "ERROR: Typhon not in registration mode");
        return 0;
    }

    CommPhase comm_phase;

    comm_phase.data_ids.clear();

    TYPH_Ghosts ghosts;
    switch (num_ghost_layers) {
    case 1: ghosts = TYPH_GHOSTS_ONE; break;
    case 2: ghosts = TYPH_GHOSTS_TWO; break;
    default:
        FAIL_WITH_LINE(err, "ERROR: Invalid num_ghost_layers");
        return 0;
    };

    if (comm.key_comm_cells == -1) {
        FAIL_WITH_LINE(err, "ERROR: Comm key set not initialised");
        return 0;
    }

    int typh_err = TYPH_Add_Phase(
            &comm_phase.typh_id,
            name.c_str(),
            ghosts,
            TYPH_PURE,
            comm.key_comm_cells,
            -1);

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Add_Phase failed");
        return 0;
    }

    CommPhaseID const comm_phase_id = comm.phases.size();
    comm.phases.push_back(comm_phase);
    return comm_phase_id;
}



void
addDataToCommPhase(
        Comm &comm,
        CommPhaseID comm_phase_id,
        DataID data_id,
        DataControl const &data,
        Error &err)
{
    if (!inRegistration()) {
        FAIL_WITH_LINE(err, "ERROR: Typhon not in registration mode");
        return;
    }

    if (comm_phase_id >= comm.phases.size()) {
        FAIL_WITH_LINE(err, "ERROR: Invalid comm_phase_id");
        return;
    }

    int const phase_id = comm.phases[comm_phase_id].typh_id;
    if (phase_id == -1) {
        FAIL_WITH_LINE(err, "ERROR: phase not initialised");
        return;
    }

    // Add ID to comm phase list
    comm.phases[comm_phase_id].data_ids.push_back(data_id);

    int const quant_id = data[data_id].getTyphonHandle();
    if (quant_id == -1) {
        FAIL_WITH_LINE(err, "ERROR: data not tagged for MPI");
        return;
    }

    // Register quant ID with Typhon
    int typh_err = TYPH_Add_Quant_To_Phase(phase_id, quant_id, -1, -1, -1, -1);

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Add_Quant_To_Phase failed");
        return;
    }
}



void
exchange(
        Comm const &comm,
        CommPhaseID comm_phase_id,
        TimerID timer_id,
        TimerControl &timers,
        Error &err)
{
    if (inRegistration()) {
        FAIL_WITH_LINE(err, "ERROR: Typhon still in registration mode");
        return;
    }

    if (comm_phase_id >= comm.phases.size()) {
        FAIL_WITH_LINE(err, "ERROR: Invalid comm_phase_id");
        return;
    }

    ScopedTimer st(timers, timer_id);

    int const phase_id = comm.phases[comm_phase_id].typh_id;

    int typh_err = TYPH_Exchange(phase_id);

    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Exchange failed");
        return;
    }
}

} // namespace comms
} // namespace bookleaf
