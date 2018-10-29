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
#ifndef BOOKLEAF_PACKAGES_HYDRO_KERNEL_GETDT_H
#define BOOKLEAF_PACKAGES_HYDRO_KERNEL_GETDT_H

#include <string>

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {

struct Error;

namespace hydro {

struct Config;

namespace kernel {

using constants::NCORN;

void
getDtCfl(
        hydro::Config const &hydro,
        int nel,
        double zcut,
        double cfl_sf,
        ConstDeviceView<unsigned char, VarDim> zdtnotreg,
        ConstDeviceView<unsigned char, VarDim> zmidlength,
        ConstDeviceView<int, VarDim>           elreg,
        ConstDeviceView<double, VarDim>        elcs2,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        DeviceView<double, VarDim>             rscratch11,
        DeviceView<double, VarDim>             rscratch12,
        double &rdt,
        int &idt,
        std::string &sdt,
        Error &err);

void
getDtDiv(
        hydro::Config const &hydro,
        int nel,
        double div_sf,
        ConstDeviceView<double, VarDim>        a1,
        ConstDeviceView<double, VarDim>        a3,
        ConstDeviceView<double, VarDim>        b1,
        ConstDeviceView<double, VarDim>        b3,
        ConstDeviceView<double, VarDim>        elvolume,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim>             scratch,
        double &rdt,
        int &idt,
        std::string &sdt);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GETDT_H
