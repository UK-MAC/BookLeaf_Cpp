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
#ifndef BOOKLEAF_RAJA_CUDA_SUPPORT
    eos.d_mat_types = new int[sizes.nmat];
    eos.d_mat_params = new double[sizes.nmat * MaterialEOS::NUM_PARAMS];

    if ((eos.d_mat_types == nullptr) || (eos.d_mat_params == nullptr)) {
        FAIL_WITH_LINE(err, "ERROR: failed to allocated EOS config");
        return;
    }

    for (int imat = 0; imat < sizes.nmat; imat++) {
        eos.d_mat_types[imat] = (int) eos.mat_eos[imat].type;
        for (int iparam = 0; iparam < MaterialEOS::NUM_PARAMS; iparam++) {
            eos.d_mat_params[imat*MaterialEOS::NUM_PARAMS+iparam] =
                eos.mat_eos[imat].params[iparam];
        }
    }
#else
    auto cuda_err = cudaMalloc((void **) &eos.d_mat_types,
            sizes.nmat * sizeof(int));
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: cudaMalloc failed");
        return;
    }

    cuda_err = cudaMalloc((void **) &eos.d_mat_params,
            sizes.nmat * MaterialEOS::NUM_PARAMS * sizeof(double));
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: cudaMalloc failed");
        return;
    }

    int *tmp_mat_types = new int[sizes.nmat];
    double *tmp_mat_params = new double[sizes.nmat * MaterialEOS::NUM_PARAMS];

    if ((tmp_mat_types == nullptr) || (tmp_mat_params == nullptr)) {
        FAIL_WITH_LINE(err, "ERROR: failed to allocated EOS config");
        return;
    }

    for (int imat = 0; imat < sizes.nmat; imat++) {
        tmp_mat_types[imat] = (int) eos.mat_eos[imat].type;
        for (int iparam = 0; iparam < MaterialEOS::NUM_PARAMS; iparam++) {
            tmp_mat_params[imat*MaterialEOS::NUM_PARAMS+iparam] =
                eos.mat_eos[imat].params[iparam];
        }
    }

    cuda_err = cudaMemcpy(eos.d_mat_types, tmp_mat_types,
            sizes.nmat * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: cudaMemcpy failed");
        return;
    }

    cuda_err = cudaMemcpy(eos.d_mat_params, tmp_mat_params,
            sizes.nmat * MaterialEOS::NUM_PARAMS * sizeof(double),
            cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: cudaMemcpy failed");
        return;
    }

    delete[] tmp_mat_types;
    delete[] tmp_mat_params;
#endif
}



void
killEOSConfig(
        EOS &eos)
{
#ifndef BOOKLEAF_RAJA_CUDA_SUPPORT
    delete[] eos.d_mat_types;
    delete[] eos.d_mat_params;
#else
    cudaFree(eos.d_mat_types);
    cudaFree(eos.d_mat_params);
#endif
}

} // namespace bookleaf
