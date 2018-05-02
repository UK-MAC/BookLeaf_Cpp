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
#ifndef BOOKLEAF_PACKAGES_SETUP_SET_INITIAL_CONDITIONS_H
#define BOOKLEAF_PACKAGES_SETUP_SET_INITIAL_CONDITIONS_H



namespace bookleaf {

struct Sizes;
struct Error;
struct GlobalConfiguration;
namespace comms { struct Comm; }

class TimerControl;
class DataControl;

namespace setup {

struct Config;

namespace driver {

void
setInitialConditions(
        setup::Config &setup_config,
        GlobalConfiguration &global,
        comms::Comm &comm,
        Sizes &sizes,
        TimerControl &timers,
        DataControl &data,
        Error &err);

} // namespace driver
} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_SET_INITIAL_CONDITIONS_H
