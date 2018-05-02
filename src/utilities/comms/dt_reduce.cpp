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
#include "utilities/comms/dt_reduce.h"

#include <typhon.h>

#include "common/dt.h"
#include "common/error.h"
#include "common/timer_control.h"



namespace bookleaf {
namespace comms {
namespace {

bool initialised = false;   //!< Has the dt reduction been initialised?

} // namespace

void
initReduceDt(
        Error &err)
{
#ifdef BOOKLEAF_MPI_DT_CONTEXT
    // Initialise dt reduction type
    int typh_err = TYPH_Add_Reduce_Dt();
    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Add_Reduce_Dt failed");
        return;
    }
#else
    // Do nothing
#endif

    initialised = true;
}



void
reduceDt(
        Dt &dt,
        TimerID timer_id,
        TimerControl &timers,
        Error &err)
{
    if (!initialised) {
        FAIL_WITH_LINE(err, "ERROR: dt reduction not initialised");
        err.iout = Error::ErrorHalt::HALT_ALL;
        return;
    }

#ifdef BOOKLEAF_MPI_DT_CONTEXT
    // Copy dt info to Typhon format
    TYPH_Dt typh_dt;
    typh_dt.rdt = dt.rdt;
    typh_dt.idt = dt.idt;
    std::copy(&dt.sdt[0], &dt.sdt[8], typh_dt.sdt);
    std::copy(&dt.mdt[0], &dt.mdt[10], typh_dt.mdt);
#else
    double rdt;
#endif

    // Perform reduction
    {
        ScopedTimer st(timers, timer_id);

#ifdef BOOKLEAF_MPI_DT_CONTEXT
        int typh_err = TYPH_Reduce_Dt(typh_dt);
#else
        int typh_err = TYPH_Reduce(&dt.rdt, nullptr, 0, &rdt, TYPH_OP_MIN);
#endif

        if (typh_err != TYPH_SUCCESS) {
            FAIL_WITH_LINE(err, "ERROR: failed to reduce dt");
            err.iout = Error::ErrorHalt::HALT_ALL;
            return;
        }
    }

#ifdef BOOKLEAF_MPI_DT_CONTEXT
    // Copy back to bookleaf format
    dt.rdt = typh_dt.rdt;
    dt.idt = typh_dt.idt;
    dt.sdt = std::string(typh_dt.sdt, 8);
    dt.mdt = std::string(typh_dt.mdt, 10);
#else
    dt.rdt = rdt;
    dt.idt = -2;
    dt.sdt = " DISABLE";
    dt.mdt = "   DISABLE";
#endif
}

} // namespace comms
} // namespace bookleaf
