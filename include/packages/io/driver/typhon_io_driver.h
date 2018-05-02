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
#ifndef BOOKLEAF_PACKAGES_IO_DRIVER_TYPHON_IO_DRIVER_H
#define BOOKLEAF_PACKAGES_IO_DRIVER_TYPHON_IO_DRIVER_H

#include "packages/io/driver/io_driver.h"



namespace bookleaf {

struct Error;
class TimerControl;

namespace io {

class TyphonIODriver : public IODriver
{
public:
    explicit TyphonIODriver() {}
    ~TyphonIODriver() {}
};

} // namespace io
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_IO_DRIVER_TYPHON_IO_DRIVER_H
