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
#ifndef BOOKLEAF_PACKAGES_INIT_DRIVER_H
#define BOOKLEAF_PACKAGES_INIT_DRIVER_H



namespace bookleaf {

struct Sizes;
struct Config;
struct Runtime;
class TimerControl;
class DataControl;
struct Error;

namespace init {
namespace driver {

/** \brief Initialise the mesh data. */
void
initMesh(
        Config &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data,
        Error &err);

/** \brief Set dummy sizes for a serial run. */
void
initSerialGhosts(
        Sizes &sizes);

/** \brief Calculate mesh connectivity. */
void
initConnectivity(
        Sizes const &sizes,
        DataControl &data);

/** \brief Calculate mesh node types. */
void
initNodeTypes(
        Sizes &sizes,
        DataControl &data);

/** \brief Calculate element ordering. */
void
initElementOrdering(
        Config const &config,
        Sizes const &sizes,
        DataControl &data);

/** \brief Initialise some mesh quantities. */
void
initElementState(
        Config const &config,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data);

} // namespace driver
} // namespace init
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_INIT_DRIVER_H
