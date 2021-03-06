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
#ifndef BOOKLEAF_UTILITIES_COMMS_DT_REDUCE_H
#define BOOKLEAF_UTILITIES_COMMS_DT_REDUCE_H



namespace bookleaf {

struct Dt;
struct Error;
enum class TimerID : int;
class TimerControl;

namespace comms {

/** \brief Initialise dt reduction. */
void
initReduceDt(
        Error &err);

/** \brief Reduce dt across ranks. */
void
reduceDt(
        Dt &dt,
        TimerID timer_id,
        TimerControl &timers,
        Error &err);

} // namespace comms
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_COMMS_DT_REDUCE_H
