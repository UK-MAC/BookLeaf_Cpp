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
#include "packages/time/driver/advance.h"

#include <cassert>

#include "common/dt.h"
#include "common/timestep.h"
#include "common/sizes.h"
#include "common/runtime.h"
#include "common/view.h"
#include "common/data_control.h"
#include "utilities/io/config.h"

#include "utilities/comms/config.h"
#ifdef BOOKLEAF_MPI_SUPPORT
#include "utilities/comms/dt_reduce.h"
#endif

#include "common/timer_control.h"
#include "packages/time/config.h"



namespace bookleaf {
namespace time {
namespace driver {
namespace {

void
setMaterial(
        io_utils::Labels const &io,
        DataID matid,
        DataControl &data,
        Dt &dt)
{
    auto mat = data[matid].chost<int, VarDim>();

    if (dt.idt >= 0) {
        int ii = mat(dt.idt);
        if (ii >= 0) {
            dt.mdt = io.smaterials[ii];

        } else {
            dt.mdt = "     MIXED";
        }

    } else {
        dt.mdt = "   UNKNOWN";
    }
}

} // namespace

void
calc(
        time::Config const &time,
        Timestep const &timestep,
        TimerID timerid,
        TimerControl &timers,
        Dt *&first,
        Dt *&current)
{
    timers.start(timerid);

    // Global maximum delta
    current = new Dt();
    first = current;
    current->rdt = time.dt_max;
    current->idt = -1;
    current->sdt = " MAXIMUM";

    // Growth of the previous delta
    current->next = new Dt();
    current = current->next;
    current->rdt = time.dt_g * timestep.dt;
    current->idt = -1;
    current->sdt = "  GROWTH";

    // Initial (first timestep) delta
    if (timestep.nstep == 0) {
        current->next = new Dt();
        current = current->next;
        current->rdt = time.dt_initial;
        current->idt = -1;
        current->sdt = " INITIAL";
    }
}



void
end(
        time::Config const &time,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data,
        Dt *&current,
        Dt *&next,
        Error &err)
{
    // Find smallest timestep
    Dt min_dt = *current;
    next = current->next;
    while (true) {
        delete current;
        if (next == nullptr) break;
        current = next;
        if (current->rdt < min_dt.rdt) min_dt = *current;
        next = current->next;
    }

    setMaterial(*time.io, DataID::IELMAT, data, min_dt);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (time.comm->nproc > 1) {
        if (min_dt.idt >= 0) {
            min_dt.idt = data[DataID::IELLOCGLOB].chost<int, VarDim>()(min_dt.idt);
        }

        comms::reduceDt(min_dt, TimerID::COLLECTIVET, timers, err);
        if (err.failed()) return;
    }
#endif // ifdef BOOKLEAF_MPI_SUPPORT

    // Set runtime data
    runtime.timestep->nstep++;
    runtime.timestep->dt = min_dt.rdt;
    runtime.timestep->sdt = min_dt.sdt;
    runtime.timestep->mdt = min_dt.mdt;
    runtime.timestep->idtel = min_dt.idt;
    runtime.timestep->time += runtime.timestep->dt;

    if (runtime.timestep->dt < time.dt_min) {
        err.iout = Error::ErrorHalt::HALT_ALL;
        err.fail("ERROR: dt = " + std::to_string(runtime.timestep->dt) +
                 " < dt_min");
    }

    timers.stop(TimerID::GETDT);
}

} // namespace driver
} // namespace time
} // namespace bookleaf
