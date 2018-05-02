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
#ifndef BOOKLEAF_STRING_UTILS_H
#define BOOKLEAF_STRING_UTILS_H

#include <string>
#include <algorithm>



namespace bookleaf {

inline void ltrim_ip(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

inline void rtrim_ip(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

inline void trim_ip(std::string &s) {
    ltrim_ip(s);
    rtrim_ip(s);
}

inline std::string ltrim(std::string s) {
    ltrim_ip(s);
    return s;
}

inline std::string rtrim(std::string s) {
    rtrim_ip(s);
    return s;
}

inline std::string trim(std::string s) {
    trim_ip(s);
    return s;
}



inline void to_upper_ip(std::string &s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
}

inline std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

} // namespace bookleaf



#endif // BOOKLEAF_STRING_UTILS_H
