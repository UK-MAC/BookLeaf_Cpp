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
#include "packages/check/driver/validate.h"

#include <iostream>

#include "common/config.h"
#include "common/runtime.h"
#include "common/data_control.h"
#include "common/error.h"

// FIXME(timrlaw): Bad include, this stuff should go in a utility
#include "infrastructure/io/output_formatting.h"

#include "packages/check/driver/tests.h"

#include "utilities/comms/config.h"



namespace bookleaf {
namespace check {
namespace driver {
namespace {

ValidationType
getValidationType(std::string sfile)
{
    // Sod
    if (sfile.find("sod") != std::string::npos) {
        return ValidationType::SOD;
    }

    return ValidationType::EMPTY;
}



double
calculateNorm(
        ValidationType validate,
        Config const &config,
        Runtime const &runtime,
        DataControl &data,
        Error &err)
{
    switch (validate) {
    case ValidationType::SOD:
        return testSod(config, runtime, data);

    default:
        FAIL_WITH_LINE(err, "ERROR: invalid validation type");
        return 0.;
    }
}



void
print(
        ValidationType validate,
        double norm)
{
    std::cout << inf::io::stripe() << "\n";
    std::cout << " VALIDATION\n";

    switch (validate) {
    case ValidationType::SOD:
        std::cout << "  Sod shocktube test case\n";
        std::cout << "   L1 norm. of density: " << std::setprecision(8) << norm << "\n";
        break;

    default:
        // Do nothing
        break;
    }
}

} // namespace

void
validate(
        std::string sfile,
        Config const &config,
        Runtime const &runtime,
        DataControl &data,
        Error &err)
{
    // Test whether a validation is required
    ValidationType const validate = getValidationType(sfile);
    if (validate == ValidationType::EMPTY) return;

    // Perform validation
    double const norm = calculateNorm(validate, config, runtime, data, err);
    if (err.failed()) return;

    // Print result
    if (config.comms->world->zmproc) {
        print(validate, norm);
    }
}

} // namespace driver
} // namespace check
} // namespace bookleaf
