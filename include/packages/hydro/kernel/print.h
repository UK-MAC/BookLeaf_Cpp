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
#ifndef BOOKLEAF_PACKAGES_HYDRO_KERNEL_PRINT_H
#define BOOKLEAF_PACKAGES_HYDRO_KERNEL_PRINT_H

#include <string>
#include <vector>
#include <iostream>
#include <memory>

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

using constants::NCORN;

struct Flags
{
    int len;

    double *vol;
    double *mass;
    double *ke;
    double *ie;
    double *density;
    double *dmn;        // Density min
    double *dmx;        // Density max
    double *pressure;
    double *pmn;        // Pressure min
    double *pmx;        // Pressure max

    Flags(int len) : len(len) {
        vol = new double[len];
        mass = new double[len];
        ke = new double[len];
        ie = new double[len];
        dmn = new double[len];
        dmx = new double[len];
        pressure = new double[len];
        pmn = new double[len];
        pmx = new double[len];
        density = new double[len];
    }

    ~Flags() {
        delete[] vol;
        delete[] mass;
        delete[] ke;
        delete[] ie;
        delete[] dmn;
        delete[] dmx;
        delete[] pressure;
        delete[] pmn;
        delete[] pmx;
        delete[] density;
    }

#ifdef BOOKLEAF_MPI_SUPPORT
    void
    reduce();
#endif
};



inline std::ostream &
operator<<(std::ostream &os, Flags const &rhs)
{
    os << "nflags = " << rhs.len << "\n\n";
    for (int i = 0; i < rhs.len; i++) {
        os << "#" << i << "\n";
        os << "\tvolume   = " << rhs.vol[i] << "\n";
        os << "\tmass     = " << rhs.mass[i] << "\n";
        os << "\tke       = " << rhs.ke[i] << "\n";
        os << "\tie       = " << rhs.ie[i] << "\n";
        os << "\tdensity  = " << rhs.density[i] << "\n";
        os << "\tdmin     = " << rhs.dmn[i] << "\n";
        os << "\tdmax     = " << rhs.dmx[i] << "\n";
        os << "\tpressure = " << rhs.pressure[i] << "\n";
        os << "\tpmin     = " << rhs.pmn[i] << "\n";
        os << "\tpmax     = " << rhs.pmx[i] << "\n";
        os << "\n";
    }

    return os;
}



void initShortPrint(Flags &flags);

void
calcShortPrint(
        int nsize,
        double dencut,
        ConstView<int, VarDim>           flag,
        ConstView<double, VarDim>        energy,
        ConstView<double, VarDim>        density,
        ConstView<double, VarDim>        mass,
        ConstView<double, VarDim>        volume,
        ConstView<double, VarDim>        pressure,
        ConstView<double, VarDim, NCORN> cnwt,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        Flags &flags);

void writeHeaderShortPrint(int nstep, double time);

void writeTableShortPrint(std::string sheader, std::string sflag,
        std::vector<std::string> const &sname, Flags const &flags);

void writeTotalShortPrint(double tot_vol, double tot_mass, double tot_ie,
        double tot_ke, double tot_pre, double tot_density);

void totalShortPrint(Flags const &flags, double &tot_vol, double &tot_mass,
        double &tot_ie, double &tot_ke, double &tot_pressure,
        double &tot_density);

void averageShortPrint(double dencut, Flags &flags);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_PRINT_H
