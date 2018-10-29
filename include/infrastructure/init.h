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
#ifndef BOOKLEAF_INFRASTRUCTURE_INIT_INIT_H
#define BOOKLEAF_INFRASTRUCTURE_INIT_INIT_H



namespace bookleaf {

namespace comms { struct Comm; }
namespace comms { struct Comms; }
struct Config;
struct Runtime;
class TimerControl;
class DataControl;
struct Error;

namespace inf {
namespace init {

/** \brief Initialise parallelism (MPI + any variants). */
void
initParallelism(
        comms::Comms &comms,
        Error &err);

#ifdef BOOKLEAF_MPI_SUPPORT
/** \brief Initialise communication phases. */
void
initCommPhases(
        comms::Comm &comm,
        DataControl &data,
        Error &err);
#endif

/** \brief Ensure configuration is valid. */
void
rationalise(
        Config &config,
        Runtime &runtime,
        Error &err);

/** \brief Main initialisation. */
void
init(
        Config &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data,
        Error &err);

} // namespace init
} // namespace inf
} // namespace bookleaf



#endif // BOOKLEAF_INFRASTRUCTURE_INIT_INIT_H
