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
#include "utilities/eos/config.h"

#include "common/sizes.h"
#include "common/error.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {

std::ostream &
operator<<(
        std::ostream &os,
        MaterialEOS const &rhs)
{
    switch (rhs.type) {
    case MaterialEOS::Type::VOID:
        os << inf::io::format_value(" EOS", "type", "VOID");
        os << inf::io::format_value(" Void pressure (p0)", "", rhs.params[0]);
        break;

    case MaterialEOS::Type::IDEAL_GAS:
        os << inf::io::format_value(" EOS", "type", "IDEAL GAS");
        os << inf::io::format_value(" Ideal gas gamma", "", rhs.params[0]);
        break;

    case MaterialEOS::Type::TAIT:
        os << inf::io::format_value(" EOS", "type", "TAIT");
        os << inf::io::format_value(" Tait a", "", rhs.params[0]);
        os << inf::io::format_value(" Tait b", "", rhs.params[1]);
        os << inf::io::format_value(" Tait rho0", "", rhs.params[2]);
        break;

    case MaterialEOS::Type::JWL:
        os << inf::io::format_value(" EOS", "type", "JWL");
        os << inf::io::format_value(" JWL omega", "", rhs.params[0]);
        os << inf::io::format_value(" JWL a", "", rhs.params[1]);
        os << inf::io::format_value(" JWL b", "", rhs.params[2]);
        os << inf::io::format_value(" JWL r1", "", rhs.params[3]);
        os << inf::io::format_value(" JWL r2", "", rhs.params[4]);
        os << inf::io::format_value(" JWL rho0", "", rhs.params[5]);
        break;

    default:
        os << inf::io::format_value("EOS", "type", "");
        break;
    }

    return os;
}



std::ostream &
operator<<(
        std::ostream &os,
        EOS const &rhs)
{
    for (int i = 0; i < (int) rhs.mat_eos.size(); i++) {
        os << "  Material: " << i << "\n";
        os << rhs.mat_eos[i];
    }

    os << inf::io::format_value("Pressure cut-off", "pcut", rhs.pcut);
    os << inf::io::format_value("Sound speed cut-off", "ccut", rhs.ccut);
    return os;
}



void
rationalise(
        EOS &eos,
        int num_materials,
        Error &err)
{
    if (eos.ccut < 0.) {
        err.fail("ERROR: ccut < 0");
        return;
    }

    if (eos.pcut < 0.) {
        err.fail("ERROR: pcut < 0");
        return;
    }

    // Helper macro to check if vector is large enough, and to shrink it
    // if it's too large
    #define CHECK_ARRAY_SIZE(V, N, ERR) { \
        if ((V).size() < (decltype((V).size())) (N)) { \
            err.fail((ERR)); \
            return; \
        } else if ((V).size() > (decltype((V).size())) (N)) { \
            (V).resize((N)); \
        } }

    CHECK_ARRAY_SIZE(eos.mat_eos, num_materials,
            "ERROR: inconsistent no. materials for eos");

    #undef CHECK_ARRAY_SIZE
}



void
initEOSConfig(
        Sizes const &sizes,
        EOS &eos,
        Error &err)
{
    // XXX Stub for extra variant EOS config init
}



void
killEOSConfig(
        EOS &eos)
{
    // XXX Stub for extra variant EOS config shutdown
}

} // namespace bookleaf
