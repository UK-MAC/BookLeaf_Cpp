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
#ifndef BOOKLEAF_PACKAGES_IO_DRIVER_IO_DRIVER_H
#define BOOKLEAF_PACKAGES_IO_DRIVER_IO_DRIVER_H

#include <string>
#include <vector>



namespace bookleaf {

struct Error;
class TimerControl;
class DataControl;
namespace comms { struct Comm; }
struct Sizes;

namespace io_utils {
struct Labels;
}

namespace io {

class IODriver
{
public:
    explicit IODriver() {}
    virtual ~IODriver() {}

    virtual void
    dump(
            std::string dir,
            io_utils::Labels const &io,
            Sizes const &sizes,
            DataControl const &data,
            comms::Comm const &comm,
            TimerControl &timers,
            Error &err) = 0;

protected:
    bool exists(std::string path);
    bool isDirectory(std::string path);
    void makeDirectory(std::string path, Error &err);
    void makeSymbolicLink(std::string target, std::string link, Error &err);
    void removeFile(std::string path, Error &err);
};

} // namespace io
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_IO_DRIVER_IO_DRIVER_H
