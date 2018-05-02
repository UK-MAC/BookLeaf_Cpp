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
#include "packages/hydro/kernel/print.h"

#include <cassert>
#include <algorithm>
#include <iomanip>
#include <numeric>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/constants.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

#ifdef BOOKLEAF_MPI_SUPPORT
void
Flags::reduce()
{
    int typh_err = TYPH_SUCCESS;
    std::unique_ptr<double[]> rval(new double[len]);

    typh_err |= TYPH_Reduce(vol, &len, 1, rval.get(), TYPH_OP_SUM);
    std::copy(&rval[0], &rval[len], vol);

    typh_err |= TYPH_Reduce(mass, &len, 1, rval.get(), TYPH_OP_SUM);
    std::copy(&rval[0], &rval[len], mass);

    typh_err |= TYPH_Reduce(ke, &len, 1, rval.get(), TYPH_OP_SUM);
    std::copy(&rval[0], &rval[len], ke);

    typh_err |= TYPH_Reduce(ie, &len, 1, rval.get(), TYPH_OP_SUM);
    std::copy(&rval[0], &rval[len], ie);

    typh_err |= TYPH_Reduce(dmn, &len, 1, rval.get(), TYPH_OP_MIN);
    std::copy(&rval[0], &rval[len], dmn);

    typh_err |= TYPH_Reduce(dmx, &len, 1, rval.get(), TYPH_OP_MAX);
    std::copy(&rval[0], &rval[len], dmx);

    typh_err |= TYPH_Reduce(pressure, &len, 1, rval.get(), TYPH_OP_SUM);
    std::copy(&rval[0], &rval[len], pressure);

    typh_err |= TYPH_Reduce(pmn, &len, 1, rval.get(), TYPH_OP_MIN);
    std::copy(&rval[0], &rval[len], pmn);

    typh_err |= TYPH_Reduce(pmx, &len, 1, rval.get(), TYPH_OP_MAX);
    std::copy(&rval[0], &rval[len], pmx);

    if (typh_err != TYPH_SUCCESS) {
        assert(false && "unhandled error");
    }
}
#endif // BOOKLEAF_MPI_SUPPORT



void
initShortPrint(Flags &flags)
{
    int const nflag = flags.len;

    // Initialise arrays
    std::fill(flags.vol, flags.vol + nflag, 0.);
    std::fill(flags.ie, flags.ie + nflag, 0.);
    std::fill(flags.density, flags.density + nflag, 0.);
    std::fill(flags.pressure, flags.pressure + nflag, 0.);
    std::fill(flags.pmx, flags.pmx + nflag, -std::numeric_limits<double>::max());
    std::fill(flags.pmn, flags.pmn + nflag, std::numeric_limits<double>::max());
    std::fill(flags.mass, flags.mass + nflag, 0.);
    std::fill(flags.ke, flags.ke + nflag, 0.);
    std::fill(flags.dmx, flags.dmx + nflag, -std::numeric_limits<double>::max());
    std::fill(flags.dmn, flags.dmn + nflag, std::numeric_limits<double>::max());
}



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
        Flags &flags)
{
    for (int ii = 0; ii < nsize; ii++) {
        int iflag = flag(ii);
        if (iflag < 0) continue;
        assert(iflag < flags.len);

        // Condition
        double const c1 = dencut * volume(ii);

        // Scatter element contributions to flag
        flags.vol[iflag] += volume(ii);
        if (mass(ii) > c1) {
            double const mas = mass(ii);
            flags.mass[iflag] += mas;

            double const en = energy(ii) * mas;
            flags.ie[iflag] += en;

            double const press = pressure(ii);
            double const w3 = mas * press;
            flags.pressure[iflag] += w3;
            flags.pmx[iflag] = std::max(flags.pmx[iflag], press);
            flags.pmn[iflag] = std::min(flags.pmn[iflag], press);

            double const dens = density(ii);
            flags.dmx[iflag] = std::max(flags.dmx[iflag], dens);
            flags.dmn[iflag] = std::min(flags.dmn[iflag], dens);

            for (int ic = 0; ic < NCORN; ic++) {
                double const u = cnu(ii, ic);
                double const v = cnv(ii, ic);
                flags.ke[iflag] += 0.5 * cnwt(ii, ic) * dens * (u*u + v*v);
            }
        }
    }
}



void
writeHeaderShortPrint(int nstep, double time)
{
    // Print header
    std::cout << "\n";
    std::cout << "  Step no. " << nstep << ", Time = " << time << "\n";
}



void
writeTableShortPrint(std::string sheader, std::string sflag,
        std::vector<std::string> const &sname, Flags const &flags)
{
    // Print header
    std::cout << "\n";
    std::cout << sheader << "\n";
    std::cout << "\n";

    std::cout << "  " << sflag;
    std::cout << "         vol";
    std::cout << "        mass";
    std::cout << "      tot_ie";
    std::cout << "      tot_ke";
    std::cout << "       press";
    std::cout << "   min press";
    std::cout << "   max press";
    std::cout << "        dens";
    std::cout << "    min dens";
    std::cout << "    max dens";
    std::cout << "\n";

    // Print table
    for (int ii = 0; ii < flags.len; ii++) {
        std::cout << std::setw(12) << sname[ii];

        std::cout << std::scientific << std::setprecision(4) << std::right;
        std::cout << std::setw(12) << flags.vol[ii];
        std::cout << std::setw(12) << flags.mass[ii];
        std::cout << std::setw(12) << flags.ie[ii];
        std::cout << std::setw(12) << flags.ke[ii];
        std::cout << std::setw(12) << flags.pressure[ii];
        std::cout << std::setw(12) << flags.pmn[ii];
        std::cout << std::setw(12) << flags.pmx[ii];
        std::cout << std::setw(12) << flags.density[ii];
        std::cout << std::setw(12) << flags.dmn[ii];
        std::cout << std::setw(12) << flags.dmx[ii];
        std::cout << "\n";
    }
}



void
writeTotalShortPrint(double tot_vol, double tot_mass, double tot_ie,
        double tot_ke, double tot_pressure, double tot_density)
{
    // Print totals
    std::cout << "\n";
    std::cout << " Total      ";
    std::cout << std::setw(12) << tot_vol;
    std::cout << std::setw(12) << tot_mass;
    std::cout << std::setw(12) << tot_ie;
    std::cout << std::setw(12) << tot_ke;
    std::cout << std::setw(12) << tot_pressure;
    std::cout << std::string(24, ' ');
    std::cout << std::setw(12) << tot_density;
    std::cout << "\n";

    std::cout << "\n";
    std::cout << "           total energy\n";
    std::cout << " internal ";
    std::cout << std::setw(12) << tot_ie << "\n";
    std::cout << " kinetic  ";
    std::cout << std::setw(12) << tot_ke << "\n";
    std::cout << " total    ";
    std::cout << std::setw(12) << tot_ie + tot_ke << "\n";
    std::cout << "\n";
}



void
totalShortPrint(Flags const &flags, double &tot_vol, double &tot_mass,
        double &tot_ie, double &tot_ke, double &tot_pressure,
        double &tot_density)
{
    // Initialise
    tot_vol = 0.;
    tot_mass = 0.;
    tot_ie = 0.;
    tot_ke = 0.;
    tot_pressure = 0.;
    tot_density = 0.;

    // Calculate totals
    for (int ii = 0; ii < flags.len; ii++) {
        tot_vol += flags.vol[ii];
        tot_mass += flags.mass[ii];
        tot_ie += flags.ie[ii];
        tot_ke += flags.ke[ii];
        tot_pressure += flags.pressure[ii] * flags.mass[ii];
    }

    tot_density = tot_mass / tot_vol;
    tot_pressure /= tot_mass;
}



void
averageShortPrint(double dencut, Flags &flags)
{
    for (int ii = 0; ii < flags.len; ii++) {
        if (flags.vol[ii] > 0.) {
            flags.density[ii] = flags.mass[ii] / flags.vol[ii];
        }

        if (flags.mass[ii] > (dencut * flags.vol[ii])) {
            flags.pressure[ii] /= flags.mass[ii];
        }
    }
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
