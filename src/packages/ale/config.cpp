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
#include "packages/ale/config.h"

#include <algorithm> // std::min_element, std::max_element

#include "common/error.h"
#include "infrastructure/io/output_formatting.h"
#include "packages/time/config.h"



namespace bookleaf {
namespace ale {

Config::Config() :
    patch_type {0},
    patch_motion {1},
    patch_ntrigger {0},
    patch_ontime {std::numeric_limits<double>::max()},
    patch_offtime {std::numeric_limits<double>::max()},
    patch_minvel {std::numeric_limits<double>::max()},
    patch_maxvel {std::numeric_limits<double>::max()},
    patch_om {1.},
    patch_trigger {{1}}
{
}



std::ostream &
operator<<(std::ostream &os, ale::Config const &rhs)
{
    os << inf::io::format_value("ALE exists", "zexist", (rhs.zexist ? "TRUE" : "FALSE"));

    if (rhs.zexist) {
        os << inf::io::format_value("Eulerian frame selected", "zeul",
                (rhs.zeul ? "TRUE" : "FALSE"));
        os << inf::io::format_value("ALE currently active", "zon",
                (rhs.zon ? "TRUE" : "FALSE"));
        os << inf::io::format_value("Advection type", "adv_type",
                (rhs.adv_type == 1 ? "Split" : "Unsplit"));
        os << inf::io::format_value("ALE timestep safety factor", "sf", rhs.sf);
    }

    return os;
}



void
rationalise(ale::Config &ale, time::Config const &time, Error &err)
{
    auto clear_arrays = [&ale]() {
        ale.patch_type.clear();
        ale.patch_motion.clear();
        ale.patch_ntrigger.clear();
        ale.patch_trigger.clear();
        ale.patch_ontime.clear();
        ale.patch_offtime.clear();
        ale.patch_minvel.clear();
        ale.patch_maxvel.clear();
        ale.patch_om.clear();
    };

    if (ale.npatch < 0 && !ale.zeul) {
        err.fail("ERROR: npatch < 0");
        return;
    }

    if (ale.zeul) {
        ale.npatch = 1;
        ale.mintime = -std::numeric_limits<double>::max();
        ale.maxtime = std::numeric_limits<double>::max();
        ale.zexist = true;
        clear_arrays();

    } else { // !ale.zeul
        if (ale.npatch > 0) {
            // Helper macro to check if vector is large enough, and to shrink it
            // if it's too large
            #define CHECK_ARRAY_SIZE(V, N, ERR) { \
                if ((V).size() < (decltype((V).size())) (N)) { \
                    err.fail((ERR)); \
                    return; \
                } else if ((V).size() > (decltype((V).size())) (N)) { \
                    (V).resize((N)); \
                } }

            CHECK_ARRAY_SIZE(ale.patch_type, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_motion, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_ntrigger, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_ontime, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_offtime, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_om, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_minvel, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");
            CHECK_ARRAY_SIZE(ale.patch_maxvel, ale.npatch,
                    "ERROR: inconsistent no. patches for ALE");

            int const ntrigger =
                *std::max_element(ale.patch_ntrigger.begin(),
                                  ale.patch_ntrigger.end());
            CHECK_ARRAY_SIZE(ale.patch_trigger, ntrigger,
                    "ERROR: inconsistent no. triggers for ALE");

            for (auto &v : ale.patch_trigger) {
                CHECK_ARRAY_SIZE(v, ale.npatch,
                        "ERROR: inconsistent no. patches for ALE");
            }

            #undef CHECK_ARRAY_SIZE

            ale.mintime = *std::min_element(ale.patch_ontime.begin(),
                    ale.patch_ontime.end());
            ale.maxtime = *std::max_element(ale.patch_offtime.begin(),
                    ale.patch_offtime.end());
            ale.zexist = (ale.mintime < time.time_start) &&
                         (ale.maxtime > time.time_end);
            ale.zexist = ale.mintime < ale.maxtime;
            if (!ale.zexist) clear_arrays();

        } else { // !(npatch > 0)
            ale.zexist = false;
            clear_arrays();
        }
    }
}

} // namespace ale
} // namespace bookleaf
