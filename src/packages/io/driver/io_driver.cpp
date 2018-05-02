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
#include "packages/io/driver/io_driver.h"

#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "common/error.h"
#include "common/cmd_args.h"



namespace bookleaf {
namespace io {

bool
IODriver::exists(std::string path)
{
    struct stat st;
    return (stat(path.c_str(), &st) == 0);
}



bool
IODriver::isDirectory(std::string path)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}



void
IODriver::makeDirectory(std::string path, Error &err)
{
    if (!exists(path)) {
        if (mkdir(path.c_str(), S_IRWXU) != 0 && errno != EEXIST) {
            err.fail("ERROR: couldn't create directory " + path);
            return;
        }

    } else if (!isDirectory(path)) {
        err.fail("ERROR: file exists with desired directory name " + path);
        return;
    }
}



void
IODriver::makeSymbolicLink(std::string target, std::string link, Error &err)
{
    if (CMD_ARGS.overwrite_dumps) removeFile(link, err);

    if (!exists(link)) {
        if (symlink(target.c_str(), link.c_str()) != 0) {
            err.fail("ERROR: couldn't create symbolic link " + link);
            return;
        }

    } else {
        err.fail("ERROR: file exists with desired symlink name " + link);
        return;
    }
}



void
IODriver::removeFile(std::string path, Error &err)
{
    if (exists(path)) {
        if (isDirectory(path)) {
            err.fail("ERROR: can't remove directory " + path);
            return;

        } else {
            if (remove(path.c_str()) != 0) {
                err.fail("ERROR: couldn't remove existing file " + path);
                return;
            }
        }
    }
}

} // namespace io
} // namespace bookleaf
