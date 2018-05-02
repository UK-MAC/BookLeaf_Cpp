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
#ifndef BOOKLEAF_COMMON_DT_H
#define BOOKLEAF_COMMON_DT_H

#include <string>
#include <iostream>



namespace bookleaf {

/** \brief A timestep value, with associated context. */
struct Dt
{
    double rdt;             //!< Dt size in seconds
    int idt;                //!< Limiting mesh element (if applicable)
    std::string sdt;        //!< Dt description
    std::string mdt;        //!< Limiting element material (if applicable)

    Dt *next = nullptr;     //!< Next Dt in linked list (if applicable)
};



inline std::ostream &
operator<<(
        std::ostream &os,
        Dt const &rhs)
{
    os << "rdt = " << rhs.rdt << "\n";
    os << "idt = " << rhs.idt << "\n";
    os << "sdt = " << rhs.sdt << "\n";
    os << "mdt = " << rhs.mdt << "\n";
    return os;
}

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_DT_H
