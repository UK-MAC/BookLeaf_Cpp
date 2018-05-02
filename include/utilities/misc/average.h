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
#ifndef BOOKLEAF_UTILITIES_MISC_AVERAGE_H
#define BOOKLEAF_UTILITIES_MISC_AVERAGE_H



namespace bookleaf {

struct Sizes;
class DataControl;
enum class DataID : int;

namespace utils {
namespace driver {

void
average(
        Sizes const &sizes,
        DataID frid,
        DataID mx1id,
        DataID mx2id,
        DataID elid,
        DataControl &data);

void
average(
        Sizes const &sizes,
        DataID frid,
        DataID mx1id,
        DataID mx2id,
        DataID el1id,
        DataID el2id,
        DataControl &data);

} // namespace driver
} // namespace utils
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_MISC_AVERAGE_H
