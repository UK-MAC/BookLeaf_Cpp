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
#include "utilities/comms/partition.h"

#include <cassert>
#include <algorithm>

#include <typhon.h>

#ifdef BOOKLEAF_PARMETIS_SUPPORT
#include <parmetis.h>
#endif

#include "common/error.h"
#include "common/constants.h"

#include "utilities/comms/config.h"



namespace bookleaf {
namespace comms {
namespace {

#ifdef BOOKLEAF_PARMETIS_SUPPORT
void
metisPartition(
        int nel,
        int nproc,
        int rank,
        MPI_Comm *comm,
        int const *_conn_data,
        View<int, VarDim> partitioning,
        Error &err)
{
    using constants::NCORN;
    using constants::NDAT;

    ConstView<int, VarDim, NDAT> conn_data(_conn_data, nel);

    std::unique_ptr<int[]> nelg(new int[nproc]);
    std::unique_ptr<int[]> sumg(new int[nproc]);

    int constexpr NCON = 1;
    int ncon = NCON;
    double constexpr UBVEC_VAL = 1.05;
    int edgecut, nndcomm, numflag, wgtflag;

    std::unique_ptr<float[]> tpgwgts(new float[NCON * nproc]);
    std::unique_ptr<float[]>   ubvec(new float[NCON]);

    std::unique_ptr<int[]> elmdist(new int[nproc + 1]);
    std::unique_ptr<int[]>    eptr(new int[nel + 1]);
    std::unique_ptr<int[]>    eind(new int[NCORN * nel]);

    int *elmwgt;

    int constexpr NUMOPTS = 3;
    int options[NUMOPTS];

    // Initialise
    edgecut = 0;
    elmwgt  = nullptr;
    nndcomm = 2;
    numflag = 0;    // 0 = c style numbering, 1 = fortran style
    wgtflag = 0;    // no element weights
    std::fill(options, options + NUMOPTS, 0);
    std::fill(tpgwgts.get(), tpgwgts.get() + NCON * nproc, 1.0 / nproc);
    std::fill(ubvec.get(), ubvec.get() + NCON, UBVEC_VAL);

    // Gather each ranks nel
    int typh_err = TYPH_Gather(&nel, nullptr, 0, nelg.get());
    if (typh_err != TYPH_SUCCESS) {
        err.fail("ERROR: TYPH_Gather failed");
        return;
    }

    elmdist[0] = 0;
    std::copy(nelg.get(), nelg.get() + nproc, elmdist.get() + 1);
    for (int i = 1; i <= nproc; i++) {
        elmdist[i] += elmdist[i-1];
    }

    // Convert global mesh into a format understood by Metis
    int j = 0;
    eptr[0] = 0;
    for (int i = 0; i < nel; i++) {
        for (int k = 3; k < NCORN + 3; k++) {
            eind[j] = conn_data(i, k);
            j++;
        }
        eptr[i+1] = j;
    }

    // Perform partitioning
    for (int iel = 0; iel < nel; iel++) {
        partitioning(iel) = rank;
    }

    int metis_err = ParMETIS_V3_PartMeshKway(
            elmdist.get(),          // How elements are distributed among procs
            eptr.get(),             // Specifies local elements
            eind.get(),             //           "
            elmwgt,                 // Element weights
            &wgtflag,               // Are element weights provided?
            &numflag,               // Indexing scheme (C or Fortran?)
            &ncon,                  // # weights per node (# balance constraints)
            &nndcomm,               // Specifies connectivity in dual graph
            &nproc,                 // Desired number of partitions
            tpgwgts.get(),          // Fraction of node weight per partition
            ubvec.get(),            // Imbalance tolerance per node weight
            options,                // Parameters to partitioner (unused)
            &edgecut,               // # edges cut (out)
            partitioning.data(),    // Local vertices
            comm);

    if (metis_err != METIS_OK) {
        FAIL_WITH_LINE(err, "ERROR: ParMETIS_V3_PartMeshKway failed");
        return;
    }

    // Sanity check that ParMETIS has not left an empty processor
    std::fill(nelg.get(), nelg.get() + nproc, 0);
    for (int iel = 0; iel < nel; iel++) {
        nelg[partitioning(iel)]++;
    }

    typh_err = TYPH_Reduce(nelg.get(), &nproc, 1, sumg.get(), TYPH_OP_SUM);
    if (typh_err != TYPH_SUCCESS) {
        err.fail("ERROR: TYPH_Reduce failed");
        return;
    }

    for (int i = 0; i < nproc; i++) {
        if (sumg[i] == 0) {
            err.fail("ERROR: METIS has given a processor no work");
            return;
        }
    }
}
#endif



#ifndef BOOKLEAF_PARMETIS_SUPPORT
void
_recursiveCoordinateBisection(
        int const dims[2],
        int x_lo,
        int y_lo,
        int x_hi,
        int y_hi,
        int npartl,
        int nparth,
        int &part,
        int *colouring)
{
    // Calculate the number of remaining partitions
    int const remaining_partitions = nparth - npartl + 1;

    // Base case, set block colouring
    if (remaining_partitions == 1) {
        part++;

        for (int i = x_lo; i < x_hi; i++) {
            for (int j = y_lo; j < y_hi; j++) {
                colouring[j * dims[0] + i] = part;
            }
        }

        return;
    }

    // Recursively divide the wider mesh dimension into n = remaining_partitions
    // sections
    int const npartmid = npartl + 1;

    int const w = x_hi - x_lo;
    int const h = y_hi - y_lo;
    int const nl = w;
    int const nk = h;

    // If the local mesh is wider than it is high...
    if (nl > nk) {
        int const nmid = nl / remaining_partitions;
        if ((npartmid - npartl) > 0) {
            _recursiveCoordinateBisection(
                    dims,
                    x_lo,
                    y_lo,
                    x_lo + nmid,
                    y_lo + nk,
                    npartl,
                    npartmid - 1,
                    part,
                    colouring);
        }

        if ((nparth - npartmid + 1) > 0) {
            _recursiveCoordinateBisection(
                    dims,
                    x_lo + nmid,
                    y_lo,
                    (x_lo + nmid) + (nl - nmid),
                    y_lo + nk,
                    npartmid,
                    nparth,
                    part,
                    colouring);
        }

    } else {
        int const nmid = nk / remaining_partitions;
        if ((npartmid - npartl) > 0) {
            _recursiveCoordinateBisection(
                    dims,
                    x_lo,
                    y_lo,
                    x_lo + nl,
                    y_lo + nmid,
                    npartl,
                    npartmid - 1,
                    part,
                    colouring);
        }

        if ((nparth - npartmid + 1) > 0) {
            _recursiveCoordinateBisection(
                    dims,
                    x_lo,
                    y_lo + nmid,
                    x_lo + nl,
                    (y_lo + nmid) + (nk - nmid),
                    npartmid,
                    nparth,
                    part,
                    colouring);
        }
    }
}

void
recursiveCoordinateBisection(
        int const dims[2],
        int nproc,
        int *colouring,
        Error &err)
{
    if (colouring == nullptr) {
        FAIL_WITH_LINE(err, "ERROR: colouring should be preallocated");
        return;
    }

    int part = -1;

    // Set initial invalid colouring for all elements
    std::fill(colouring, colouring + (dims[0] * dims[1]), part);

    // Perform recursion
    _recursiveCoordinateBisection(
            dims,
            0,
            0,
            dims[0],
            dims[1],
            0,
            nproc - 1,
            part,
            colouring);

    // Check all elements are coloured, and all processors have been assigned
    // work
    bool *procs_seen = new bool[nproc];
    std::fill(procs_seen, procs_seen + nproc, false);

    for (int i = 0; i < dims[0] * dims[1]; i++) {

        // Check colouring is valid
        if (!(colouring[i] >= 0 && colouring[i] < nproc)) {
            FAIL_WITH_LINE(err, "ERROR: element did not receive a valid colour");
            return;
        }

        // Mark processor as having some work
        procs_seen[colouring[i]] = true;
    }

    if (!std::all_of(procs_seen, procs_seen + nproc, [](bool v) { return v; })) {
        FAIL_WITH_LINE(err, "ERROR: processor did not receive any work");
        return;
    }

    delete[] procs_seen;
}
#endif // !BOOKLEAF_PARMETIS_SUPPORT

} // namespace

void
initialPartition(
        int const dims[2],
        Comm const &comm,
        int &side,
        int slice[2],
        int &nel)
{
    side = dims[1] >= dims[0] ? 1 : 0;

    // Split the mesh along its longest side
    int const ncol = std::max(dims[0], dims[1]);
    int const col_per_proc = ncol / comm.nproc;
    int const excess = comm.nproc * (col_per_proc + 1) - ncol;

    if (comm.rank < excess) {
        slice[0] = comm.rank * col_per_proc;
        slice[1] = slice[0] + col_per_proc;

    } else {
        slice[0] = excess * col_per_proc + (comm.rank - excess) * (col_per_proc + 1);
        slice[1] = slice[0] + (col_per_proc + 1);
    }

    // Compute local number of elements
    nel = dims[(side == 1 ? 0 : 1)] * (slice[1] - slice[0]);
}



void
improvePartition(
#ifndef BOOKLEAF_PARMETIS_SUPPORT
        int const dims[2],
        int const side,
        int const slice[2],
#else
        int const dims[2] __attribute__((unused)),
        int const side __attribute__((unused)),
        int const slice[2] __attribute__((unused)),
#endif
        int const nel,
        Comm &comm,
#ifdef BOOKLEAF_PARMETIS_SUPPORT
        int const *connectivity,
#else
        int const *connectivity __attribute__((unused)),
#endif
        View<int, VarDim> partitioning,
        Error &err)
{
#ifdef BOOKLEAF_PARMETIS_SUPPORT
    // Use the connectivity to compute a good partitioning in parallel.
    metisPartition(nel, comm.nproc, comm.rank, &comm.comm, connectivity,
            partitioning, err);
    if (err.failed()) {
        return;
    }
#else
    // If ParMETIS isn't available, just use RCB. Each processor needs to
    // partition the entire mesh.
    int *colouring = new int[dims[0] * dims[1]];

    recursiveCoordinateBisection(dims, comm.nproc, colouring, err);
    if (err.failed()) {
        delete[] colouring;
        return;
    }

    // Figure out the assignment for our slice of connectivity.
    int iel = 0;
    if (side == 1) {
        for (int i = slice[0]; i < slice[1]; i++) {
            for (int j = 0; j < dims[0]; j++) {
                assert(iel < nel);
                partitioning(iel) = colouring[(dims[0] * slice[0]) + iel];
                iel++;
            }
        }

    } else {
        for (int i = 0; i < dims[1]; i++) {
            for (int j = slice[0]; j < slice[1]; j++) {
                assert(iel < nel);
                partitioning(iel) = colouring[(j * dims[0]) + slice[0] + iel];
                iel++;
            }
        }
    }

    if (iel != nel) {
        FAIL_WITH_LINE(err, "ERROR: mismatched element count in partitioning");
        delete[] colouring;
        return;
    }

    delete[] colouring;
#endif
}

} // namespace comms
} // namespace bookleaf
