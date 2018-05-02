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
#ifndef BOOKLEAF_PACKAGES_CHECK_DRIVER_VALIDATE_H
#define BOOKLEAF_PACKAGES_CHECK_DRIVER_VALIDATE_H

#include <string>



namespace bookleaf {

struct Config;
struct Runtime;
class DataControl;
struct Error;

namespace check {

/** \brief Select problem to validate. */
enum class ValidationType : int
{
    EMPTY = 0,
    SOD
};

namespace driver {

/**
 * \brief Given the name of the input deck, determine if a validator exists and
 *        run it.
 */
void
validate(
        std::string sfile,
        Config const &config,
        Runtime const &runtime,
        DataControl &data,
        Error &err);

} // namespace driver
} // namespace check
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_CHECK_DRIVER_VALIDATE_H
