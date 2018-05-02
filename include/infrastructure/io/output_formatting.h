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
/*
 * Helper functions for formatting variables and descriptions in a consistent
 * style.
 */
#ifndef BOOKLEAF_INFRASTRUCTURE_IO_OUPUT_FORMATTING_H
#define BOOKLEAF_INFRASTRUCTURE_IO_OUPUT_FORMATTING_H

#include <string>
#include <sstream>
#include <iomanip>



namespace bookleaf {
namespace inf {
namespace io {

int constexpr OUTPUT_WIDTH      = 132;
int constexpr DESCRIPTION_WIDTH = 83;
int constexpr VALUE_WIDTH       = OUTPUT_WIDTH - DESCRIPTION_WIDTH;



inline std::string
stripe()
{
    std::ostringstream oss;
    oss << ' ';
    oss << std::string(OUTPUT_WIDTH-1, '#');
    return oss.str();
}



template <typename T>
inline std::string
format_value(std::string long_name,
             std::string short_name,
             T           value)
{
    // Pad between the long and short variable names with spaces
    std::string description_padding(
            DESCRIPTION_WIDTH - (4 + long_name.size() + short_name.size()),
            ' ');

    std::ostringstream oss;
    oss << std::setw(DESCRIPTION_WIDTH);
    oss << "  " + 
           long_name +
           (!long_name.empty() ? ":" : " ") +
           description_padding +
           short_name +
           " ";

    oss << std::right;
    oss << std::setw(VALUE_WIDTH);
    oss << value;

    oss << "\n";
    return oss.str();
}

// Specialisation for printing doubles in scientific notation
template <>
inline std::string
format_value(std::string long_name,
             std::string short_name,
             double      value)
{
    // Pad between the long and short variable names with spaces
    std::string description_padding(
            DESCRIPTION_WIDTH - (4 + long_name.size() + short_name.size()),
            ' ');

    std::ostringstream oss;
    oss << std::setw(DESCRIPTION_WIDTH);
    oss << "  " + 
           long_name +
           (!long_name.empty() ? ":" : " ") +
           description_padding +
           short_name +
           " ";

    oss << std::right;
    oss << std::setw(VALUE_WIDTH);
    oss << std::scientific;
    oss << value;

    oss << "\n";
    return oss.str();
}

} // namespace io
} // namespace inf
} // namespace bookleaf



#endif // BOOKLEAF_INFRASTRUCTURE_IO_OUPUT_FORMATTING_H
