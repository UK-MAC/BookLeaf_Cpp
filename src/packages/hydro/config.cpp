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
#include "packages/hydro/config.h"

#include <algorithm> // std::any_of

#include "common/sizes.h"
#include "common/error.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {
namespace hydro {

Config::Config()
{
}



std::ostream &
operator<<(std::ostream &os, Config const &rhs)
{
    os << inf::io::format_value("Linear artificial viscosity", "cvisc1", rhs.cvisc1);
    os << inf::io::format_value("Quadratic artificial viscosity", "cvisc2", rhs.cvisc2);
    os << inf::io::format_value("Hourglass filters", "zhg", (rhs.zhg ? "TRUE" : "FALSE"));
    os << inf::io::format_value("Sub-zonal pressures", "zsp", (rhs.zsp ? "TRUE" : "FALSE"));
    os << inf::io::format_value("CFL safety factor", "cfl_sf", rhs.cfl_sf);
    os << inf::io::format_value("Divergence safety factor", "div_sf", rhs.div_sf);

    os << inf::io::format_value("Mid-length of projection for CFL length scale",
            "zmidlength", "");
    os << inf::io::format_value("Exclude region from CFL calculation", "zdtnotreg", "");

    if (rhs.zhg) {
        os << "\n";
        os << "  region         kappareg\n";
        for (int i = 0; i < (int) rhs.kappareg.size(); i++) {
            os << "       " << i << " " << rhs.kappareg[i] << "\n";;
        }
    }

    if (rhs.zsp) {
        os << "\n";
        os << "  region        pmeritreg\n";
        for (int i = 0; i < (int) rhs.pmeritreg.size(); i++) {
            os << "       " << i << " " << rhs.pmeritreg[i] << "\n";;
        }
    }

    os << "\n";
    os << "  region mid-length\n";
    for (int i = 0; i < (int) rhs.zmidlength.size(); i++) {
        if (rhs.zmidlength[i]) {
            os << "       " << i << "       TRUE\n";
        } else {
            os << "       " << i << "      FALSE\n";
        }
    }

    os << "\n";
    os << "  region  cfl calc.\n";
    for (int i = 0; i < (int) rhs.zdtnotreg.size(); i++) {
        if (!rhs.zdtnotreg[i]) {
            os << "       " << i << "       TRUE\n";
        } else {
            os << "       " << i << "      FALSE\n";
        }
    }

    return os;
}



void
rationalise(Config &hydro, int num_regions, Error &err)
{
    if (hydro.cvisc1 < 0.) {
        err.fail("ERROR: cvisc1 < 0");
        return;
    }

    if (hydro.cvisc2 < 0.) {
        err.fail("ERROR: cvisc2 < 0");
        return;
    }

    if (hydro.cfl_sf < 0.) {
        err.fail("ERROR: cfl_sf < 0");
        return;
    }

    if (hydro.div_sf < 0.) {
        err.fail("ERROR: div_sf < 0");
        return;
    }

    // Helper macro to check if vector is large enough, and to shrink it if it's
    // too large. Sets default value for expanded vectors.
    #define CHECK_ARRAY_SIZE(V, N, DEFAULT) { \
        if ((V).size() < (decltype((V).size())) (N)) { \
            (V).resize((N), (DEFAULT)); \
        } else if ((V).size() > (decltype((V).size())) (N)) { \
            (V).resize((N)); \
        } }

    CHECK_ARRAY_SIZE(hydro.kappareg, num_regions, hydro.kappaall);
    hydro.zhg = std::any_of(hydro.kappareg.begin(), hydro.kappareg.end(),
            [](double v) { return v > 0.; });

    CHECK_ARRAY_SIZE(hydro.pmeritreg, num_regions, hydro.pmeritall);
    hydro.zsp = std::any_of(hydro.pmeritreg.begin(), hydro.pmeritreg.end(),
            [](double v) { return v > 0.; });

    CHECK_ARRAY_SIZE(hydro.zdtnotreg, num_regions, false);
    CHECK_ARRAY_SIZE(hydro.zmidlength, num_regions, false);

    #undef CHECK_ARRAY_SIZE
}



void
initHydroConfig(
        Sizes const &sizes __attribute__((unused)),
        hydro::Config &hydro __attribute__((unused)),
        Error &err __attribute__((unused)))
{
    // XXX Stub for extra variant hydro config init
}



void
killHydroConfig(
        hydro::Config &hydro __attribute__((unused)))
{
    // XXX Stub for extra variant hydro config shutdown
}

} // namespace hydro
} // namespace bookleaf
