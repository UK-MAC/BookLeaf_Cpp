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
#ifndef BOOKLEAF_COMMON_ERROR_H
#define BOOKLEAF_COMMON_ERROR_H

#include <string>



namespace bookleaf {

struct Error
{
    enum ErrorState : int
    {
        SUCCESS = 0,
        FAILURE = -1
    };

    enum ErrorHalt : int
    {
        HALT_ALL = 1,
        HALT_SINGLE = 2
    };

    ErrorState ierr = ErrorState::SUCCESS;
    ErrorHalt iout  = ErrorHalt::HALT_SINGLE;
    std::string serr;

    bool success() const { return ierr == ErrorState::SUCCESS; }
    bool failed() const { return ierr == ErrorState::FAILURE; }

    void succeed() {
        ierr = ErrorState::SUCCESS;
        serr = "";
    }

    // Set FAILURE state and message
    void fail(std::string serr) {
        ierr = ErrorState::FAILURE;
        this->serr = serr;
    }
};

// Mark a failure and include the line # in the error message
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define FAIL_WITH_LINE(err, serr) (err).fail(std::string(serr) + \
        " [" TOSTRING(__FILE__) ":" TOSTRING(__LINE__) "]")



struct Config;
struct Runtime;
class TimerControl;
class DataControl;

/** \brief Halt the program according to error state */
void
halt(
        Config const &config,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data,
        Error err);

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_ERROR_H
