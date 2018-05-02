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
#include "packages/time/driver/utils.h"

#include <iostream>
#include <iomanip>

#include "common/timer_control.h"
#include "utilities/comms/config.h"
#include "common/timestep.h"



namespace bookleaf {
namespace time {
namespace driver {

void
printCycle(
        int nel,
        comms::Comm const &comm,
        Timestep const &timestep,
        Timer tcycle,
        TimerID timerid,
        TimerControl &timers)
{
    // Binding
    if (comm.zmproc) {
        ScopedTimer st(timers, timerid);

        float const grind = timers.getGrind(nel, tcycle);
        float const timer = timers.getCount(tcycle);

        // Write out cycle information
        std::cout << " step=";
        std::cout << std::setw(7) << timestep.nstep;
        std::cout << "  el=";
        std::cout << std::setw(9) << timestep.idtel;
        std::cout << " ";
        std::cout << std::setw(10) << timestep.mdt;
        std::cout << " dt=";
        std::cout << std::setprecision(9) << std::setw(16) << timestep.dt;
        std::cout << "  time=";
        std::cout << std::setprecision(9) << std::setw(16) << timestep.time;
        std::cout << "  grind=";
        std::cout << std::setprecision(1) << std::setw(8) << grind;
        std::cout << "  timer=";
        std::cout << std::setprecision(9) << std::setw(16) << timer;
        std::cout << " s ";
        std::cout << std::setw(8) << timestep.sdt;
        std::cout << "\n";
    }
}



bool
finish(
        Timestep const &timestep,
        time::Config const &time)
{
    return timestep.time >= time.time_end;
}

} // namespace driver
} // namespace time
} // namespace bookleaf
