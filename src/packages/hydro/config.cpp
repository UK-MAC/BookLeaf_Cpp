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

#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
#include <cuda_runtime.h>
#endif

#include "common/sizes.h"
#include "common/error.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {
namespace hydro {

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
        Sizes const &sizes,
        hydro::Config &hydro,
        Error &err)
{
    // Copy hydro parameters to device
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    // Initialise device-side memory
    cudaMalloc(&hydro._dev_kappareg, sizeof(double) * sizes.nreg);
    cudaMalloc(&hydro._dev_pmeritreg, sizeof(double) * sizes.nreg);
    cudaMalloc(&hydro._dev_zdtnotreg, sizeof(unsigned char) * sizes.nreg);
    cudaMalloc(&hydro._dev_zmidlength, sizeof(unsigned char) * sizes.nreg);

    hydro.dev_kappareg =
        DeviceView<double, VarDim>(hydro._dev_kappareg, sizes.nreg);
    hydro.dev_pmeritreg =
        DeviceView<double, VarDim>(hydro._dev_pmeritreg, sizes.nreg);
    hydro.dev_zdtnotreg =
        DeviceView<unsigned char, VarDim>(hydro._dev_zdtnotreg, sizes.nreg);
    hydro.dev_zmidlength =
        DeviceView<unsigned char, VarDim>(hydro._dev_zmidlength, sizes.nreg);

    // Fill device-side memory
    cudaMemcpy(hydro._dev_kappareg, hydro.kappareg.data(),
            sizeof(double) * sizes.nreg, cudaMemcpyHostToDevice);
    cudaMemcpy(hydro._dev_pmeritreg, hydro.pmeritreg.data(),
            sizeof(double) * sizes.nreg, cudaMemcpyHostToDevice);
    cudaMemcpy(hydro._dev_zdtnotreg, hydro.zdtnotreg.data(),
            sizeof(unsigned char) * sizes.nreg, cudaMemcpyHostToDevice);
    cudaMemcpy(hydro._dev_zmidlength, hydro.zmidlength.data(),
            sizeof(unsigned char) * sizes.nreg, cudaMemcpyHostToDevice);
#else
    // Initialise host-side memory
    hydro._dev_kappareg = new double[sizes.nreg];
    hydro._dev_pmeritreg = new double[sizes.nreg];
    hydro._dev_zdtnotreg = new unsigned char[sizes.nreg];
    hydro._dev_zmidlength = new unsigned char[sizes.nreg];

    hydro.dev_kappareg =
        DeviceView<double, VarDim>(hydro._dev_kappareg, sizes.nreg);
    hydro.dev_pmeritreg =
        DeviceView<double, VarDim>(hydro._dev_pmeritreg, sizes.nreg);
    hydro.dev_zdtnotreg =
        DeviceView<unsigned char, VarDim>(hydro._dev_zdtnotreg, sizes.nreg);
    hydro.dev_zmidlength =
        DeviceView<unsigned char, VarDim>(hydro._dev_zmidlength, sizes.nreg);

    // Fill host-side memory
    memcpy(hydro._dev_kappareg, hydro.kappareg.data(),
            sizeof(double) * sizes.nreg);
    memcpy(hydro._dev_pmeritreg, hydro.pmeritreg.data(),
            sizeof(double) * sizes.nreg);
    memcpy(hydro._dev_zdtnotreg, hydro.zdtnotreg.data(),
            sizeof(unsigned char) * sizes.nreg);
    memcpy(hydro._dev_zmidlength, hydro.zmidlength.data(),
            sizeof(unsigned char) * sizes.nreg);
#endif

}



void
killHydroConfig(
        hydro::Config &hydro)
{
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    if (hydro._dev_kappareg) {
        cudaFree(hydro._dev_kappareg);
        hydro._dev_kappareg = nullptr;
    }

    if (hydro._dev_pmeritreg) {
        cudaFree(hydro._dev_pmeritreg);
        hydro._dev_pmeritreg = nullptr;
    }

    if (hydro._dev_zdtnotreg) {
        cudaFree(hydro._dev_zdtnotreg);
        hydro._dev_zdtnotreg = nullptr;
    }

    if (hydro._dev_zmidlength) {
        cudaFree(hydro._dev_zmidlength);
        hydro._dev_zmidlength = nullptr;
    }
#else
    if (hydro._dev_kappareg) {
        delete[] hydro._dev_kappareg;
        hydro._dev_kappareg = nullptr;
    }

    if (hydro._dev_pmeritreg) {
        delete[] hydro._dev_pmeritreg;
        hydro._dev_pmeritreg = nullptr;
    }

    if (hydro._dev_zdtnotreg) {
        delete[] hydro._dev_zdtnotreg;
        hydro._dev_zdtnotreg = nullptr;
    }

    if (hydro._dev_zmidlength) {
        delete[] hydro._dev_zmidlength;
        hydro._dev_zmidlength = nullptr;
    }
#endif
}

} // namespace hydro
} // namespace bookleaf
