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
#ifndef BOOKLEAF_PACKAGES_HYDRO_CONFIG_H
#define BOOKLEAF_PACKAGES_HYDRO_CONFIG_H

#include <vector>
#include <iostream>
#include <memory>



namespace bookleaf {

struct Sizes;
struct Error;
namespace comms { struct Comm; }
struct GlobalConfiguration;
namespace io_utils { struct Labels; }
struct EOS;

namespace hydro {

struct Config
{
    double cvisc1 = 0.5;
    double cvisc2 = 0.7;
    double cfl_sf = 0.5;
    double div_sf = 0.25;
    bool zhg = false;
    bool zsp = false;

    double kappaall = 0.;
    double pmeritall = 0.;

    std::vector<double> kappareg;
    std::vector<double> pmeritreg;
    std::vector<unsigned char> zdtnotreg;
    std::vector<unsigned char> zmidlength;

    std::shared_ptr<comms::Comm>         comm;
    std::shared_ptr<GlobalConfiguration> global;
    std::shared_ptr<io_utils::Labels>    io;
    std::shared_ptr<EOS>                 eos;

    Config();
};



std::ostream &
operator<<(std::ostream &os, hydro::Config const &rhs);

void
rationalise(hydro::Config &hydro, int num_regions, Error &err);

void
initHydroConfig(
        Sizes const &sizes,
        hydro::Config &hydro,
        Error &err);

void
killHydroConfig(
        hydro::Config &hydro);

} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_CONFIG_H
