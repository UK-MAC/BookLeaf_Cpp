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
#include "packages/setup/distribute_mesh.h"

#include <cassert>
#include <fstream>

#include <typhon.h>

#include "common/config.h"
#include "common/sizes.h"
#include "common/constants.h"
#include "packages/ale/config.h"
#include "common/data_control.h"
#include "utilities/comms/config.h"



namespace bookleaf {

/**
 * @fn      distributeMesh
 * @brief   Takes mesh connectivity data and partition data, and distributes the
 *          mesh amongst participating processors.
 *
 * @param [in]      conn_data
 * @param [in]      conn_dims
 * @param [in]      part
 * @param [in]      config
 * @param [inout]   sizes
 * @param [inout]   data
 * @param [out]     err
 */
void
distributeMesh(int const *conn_data, int const *conn_dims, int const *part,
        Config const &config, Sizes &sizes, DataControl &data, Error &err)
{
    using constants::NCORN;

    // Number of ghost layers per proc
    int const nlay = config.ale->zexist ? 2 : 1;

    int *layer_nel = new int[nlay + 1];
    int *layer_nnd = new int[nlay + 1];
    int total_nel = 0;
    int total_nnd = 0;
    int *el_loc_glob;
    int *nd_loc_glob;
    int *el_region;
    int *el_material;
    int *el_nd;
    int *el_owner;
    int *nd_owner;
    int typh_err = TYPH_Distribute_Mesh(
            conn_dims[1],
            sizes.nel,
            sizes.nnd,
            NCORN,
            nlay,
            conn_data,
            part,
            layer_nel,
            layer_nnd,
            &total_nel,
            &total_nnd,
            &el_loc_glob,
            &nd_loc_glob,
            &el_region,
            &el_material,
            &el_nd,
            &el_owner,
            &nd_owner);
    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Distribute_Mesh failed");
        return;
    }

    sizes.nel  = layer_nel[0];
    sizes.nnd  = layer_nnd[0];
    sizes.nel2 = total_nel;
    sizes.nnd2 = total_nnd;

    // Set mesh data
    data.setMesh(sizes);
    if (err.failed()) return;

    // Set typhon data
    data.setTyphon(sizes);
    if (err.failed()) return;

    auto ellocglob = data[DataID::IELLOCGLOB].host<int, VarDim>();
    auto ndlocglob = data[DataID::INDLOCGLOB].host<int, VarDim>();
    auto elreg     = data[DataID::IELREG].host<int, VarDim>();
    auto elmat     = data[DataID::IELMAT].host<int, VarDim>();
    auto elnd      = data[DataID::IELND].host<int, VarDim, NCORN>();

    // Copy data into bookleaf managed arrays
    {
        ConstView<int, VarDim> tmp_ellocglob(el_loc_glob, total_nel);
        ConstView<int, VarDim> tmp_ndlocglob(nd_loc_glob, total_nnd);

        for (int iel = 0; iel < total_nel; iel++) {
            ellocglob(iel) = tmp_ellocglob(iel);
        }

        for (int ind = 0; ind < total_nnd; ind++) {
            ndlocglob(ind) = tmp_ndlocglob(ind);
        }
    }

    delete[] el_loc_glob;
    delete[] nd_loc_glob;

    {
        ConstView<int, VarDim>        tmp_elreg(el_region, total_nel);
        ConstView<int, VarDim>        tmp_elmat(el_material, total_nel);
        ConstView<int, VarDim, NCORN> tmp_elnd(el_nd, total_nel);

        for (int iel = 0; iel < total_nel; iel++) {
            elreg(iel) = tmp_elreg(iel);
            elmat(iel) = tmp_elmat(iel);

            for (int icn = 0; icn < NCORN; icn++) {
                elnd(iel, icn) = tmp_elnd(iel, icn);
            }
        }
    }

    delete[] el_region;
    delete[] el_material;
    delete[] el_nd;

    switch (nlay) {
    case 1:
        sizes.nel1 = sizes.nel2;
        sizes.nnd1 = sizes.nnd2;
        break;

    case 2:
        sizes.nel1 = sizes.nel + layer_nel[1];
        sizes.nnd1 = sizes.nnd + layer_nnd[1];
        break;

    default:
        FAIL_WITH_LINE(err, "ERROR: incorrect value for nlay");
        return;
    }

    // Setup Typhon communication
    int *nel_total = new int[nlay+1];
    int *nnd_total = new int[nlay+1];
    nel_total[0] = layer_nel[0];
    nnd_total[0] = layer_nnd[0];
    for (int l = 1; l <= nlay; l++) {
        nel_total[l] = nel_total[l-1] + layer_nel[l];
        nnd_total[l] = nnd_total[l-1] + layer_nnd[l];
    }

    int partition_id;
    typh_err = TYPH_Set_Partition_Info(partition_id, TYPH_SHAPE_QUAD, nlay,
            nel_total, nnd_total, el_owner, nd_owner, ellocglob.data(),
            ndlocglob.data(), elnd.data());
    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Set_Partition_Info failed");
        return;
    }

    delete[] nel_total;
    delete[] nnd_total;
    delete[] el_owner;
    delete[] nd_owner;

    int key_comm_cells;
    typh_err = TYPH_Create_Key_Set(TYPH_KEYTYPE_CELL, 1, nlay, partition_id,
            key_comm_cells);
    if (typh_err != TYPH_SUCCESS) {
        FAIL_WITH_LINE(err, "ERROR: TYPH_Create_Key_Set failed");
        return;
    }
    config.comms->spatial->key_comm_cells = key_comm_cells;
}

} // namespace bookleaf
