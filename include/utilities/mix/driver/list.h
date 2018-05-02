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
#ifndef BOOKLEAF_UTILITIES_MIX_DRIVER_LIST_H
#define BOOKLEAF_UTILITIES_MIX_DRIVER_LIST_H



namespace bookleaf {

struct Sizes;
class DataControl;
struct Error;

namespace mix {
namespace driver {

double constexpr INCR = 0.05;

//int addEl(int iel, Sizes &sizes, DataControl &data, Error &err);

//int addCp(int imix, Sizes &sizes, DataControl &data, Error &err);

void
resizeMx(
        Sizes &sizes,
        DataControl &data,
        int nsz,
        Error &err);

void
resizeCp(
        Sizes &sizes,
        DataControl &data,
        int nsz,
        Error &err);

void
flatten(
        Sizes const &sizes,
        DataControl &data);

} // namespace driver
} // namespace mix
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_MIX_DRIVER_LIST_H
