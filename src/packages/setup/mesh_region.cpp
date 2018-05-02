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
#include "packages/setup/mesh_region.h"

#include <cassert>
#include <algorithm> // std::count_if
#include <fstream>
#include <bitset>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/error.h"
#include "common/sizes.h"
#include "infrastructure/io/output_formatting.h"
#include "common/data_control.h"



namespace bookleaf {
namespace setup {

void
printMeshRegions(std::vector<MeshRegion> const &mesh_regions,
        Sizes const &sizes)
{
    int const num_mesh_regions = mesh_regions.size();

    int const num_mesh_materials =
        std::count_if(mesh_regions.begin(), mesh_regions.end(),
            [](MeshRegion const &mr) {
                return mr.material != -1;
        });

    std::cout <<
        inf::io::format_value("Number of mesh regions", "nreg", num_mesh_regions);
    std::cout <<
        inf::io::format_value("Number of mesh materials", "nmat", num_mesh_materials);
    std::cout << inf::io::format_value("Total number of elements", "nel", sizes.nel);
    std::cout << inf::io::format_value("Total number of nodes", "nnd", sizes.nnd);

    for (int i = 0; i < (int) mesh_regions.size(); i++) {
        MeshRegion const &mr = mesh_regions[i];
        std::cout << "  Region: " << i << "\n";
        std::cout << mr;
    }

}



std::ostream &
operator<<(std::ostream &os, MeshRegion const &rhs)
{
    MeshRegion const &mr = rhs;

    os << inf::io::format_value(" Material index", "region_material", mr.material);

    switch (mr.type) {
    case MeshRegion::Type::LIN1:
        os << inf::io::format_value(" Region type", "region_typ", "LIN1"); break;
    case MeshRegion::Type::LIN2:
        os << inf::io::format_value(" Region type", "region_typ", "LIN2"); break;
    case MeshRegion::Type::EQUI:
        os << inf::io::format_value(" Region type", "region_typ", "EQUI"); break;
    case MeshRegion::Type::USER:
        os << inf::io::format_value(" Region type", "region_typ", "USER"); break;
    default:
        os << inf::io::format_value(" Region type", "region_typ", ""); break;
    }

    os << inf::io::format_value(" Region dimensions", "region_dims",
            std::to_string(mr.dims[0]) + "x" + std::to_string(mr.dims[1]));
    os << inf::io::format_value(" Convergence tolerance", "region_tol", mr.tol);
    os << inf::io::format_value(" Convergence scaling factor", "region_om", mr.om);
    os << inf::io::format_value(" Number of iterations", "no_it", mr.no_it);

    for (int j = 0; j < (int) mr.sides.size(); j++) {
        MeshRegion::Side const &mrs = mr.sides[j];

        os << "   Side: " << j << "\n";
        os << inf::io::format_value("  Number of segments", "", mrs.segments.size());

        for (int k = 0; k < (int) mrs.segments.size(); k++) {
            MeshRegion::Side::Segment const &mrss = mrs.segments[k];

            os << "    Segment: " << k << "\n";

            std::string pos;
            switch (mrss.type) {
            case MeshRegion::Side::Segment::Type::LINE:
                os << inf::io::format_value("   Segment type", "", "LINE");

                pos = "(" + std::to_string(mrss.pos[0]) + ", " +
                            std::to_string(mrss.pos[1]) + ") - " +
                      "(" + std::to_string(mrss.pos[2]) + ", " +
                            std::to_string(mrss.pos[3]) + ")";
                os << inf::io::format_value("   Segment position", "", pos);
                break;

            case MeshRegion::Side::Segment::Type::ARC_C:
                os << inf::io::format_value("   Segment type", "", "ARC_C");

                pos = "(" + std::to_string(mrss.pos[0]) + ", " +
                            std::to_string(mrss.pos[1]) + ") - " +
                      "(" + std::to_string(mrss.pos[2]) + ", " +
                            std::to_string(mrss.pos[3]) + ")";
                os << inf::io::format_value("   Segment position", "", pos);

                pos = "about (" + std::to_string(mrss.pos[4]) + ", " +
                                  std::to_string(mrss.pos[5]) + ")";
                os << inf::io::format_value("", "", pos);
                break;

            case MeshRegion::Side::Segment::Type::ARC_A:
                os << inf::io::format_value("   Segment type", "", "ARC_A");

                pos = "(" + std::to_string(mrss.pos[0]) + ", " +
                            std::to_string(mrss.pos[1]) + ") - " +
                      "(" + std::to_string(mrss.pos[2]) + ", " +
                            std::to_string(mrss.pos[3]) + ")";
                os << inf::io::format_value("   Segment position", "", pos);

                pos = "about (" + std::to_string(mrss.pos[4]) + ", " +
                                  std::to_string(mrss.pos[5]) + ")";
                os << inf::io::format_value("", "", pos);
                break;

            case MeshRegion::Side::Segment::Type::POINT:
                os << inf::io::format_value("   Segment type", "", "POINT");

                pos = "(" + std::to_string(mrss.pos[0]) + ", " +
                            std::to_string(mrss.pos[1]) + ")";
                os << inf::io::format_value("   Segment position", "", pos);
                break;

            case MeshRegion::Side::Segment::Type::LINK:
                os << inf::io::format_value("   Segment type", "", "LINK");
                break;

            default:
                os << inf::io::format_value("   Segment type", "", "");
                break;
            }

            switch (mrss.bc) {
            case MeshRegion::Side::Segment::BoundaryCondition::SLIPX:
                os << inf::io::format_value("   Segment boundary condition", "",
                        "SLIPX"); break;
            case MeshRegion::Side::Segment::BoundaryCondition::SLIPY:
                os << inf::io::format_value("   Segment boundary condition", "",
                        "SLIPY"); break;
            case MeshRegion::Side::Segment::BoundaryCondition::WALL:
                os << inf::io::format_value("   Segment boundary condition", "",
                        "WALL"); break;
            case MeshRegion::Side::Segment::BoundaryCondition::TRANS:
                os << inf::io::format_value("   Segment boundary condition", "",
                        "TRANS"); break;
            case MeshRegion::Side::Segment::BoundaryCondition::OPEN:
                os << inf::io::format_value("   Segment boundary condition", "",
                        "OPEN"); break;
            case MeshRegion::Side::Segment::BoundaryCondition::FREE:
                os << inf::io::format_value("   Segment boundary condition", "",
                        "FREE"); break;
            }
        }
    }

    return os;
}



void
rationaliseMeshRegions(std::vector<MeshRegion> &mesh_regions,
        Sizes const &sizes, Error &err)
{
    if (mesh_regions.size() < 1) {
        err.fail("ERROR: no mesh regions specified");
        return;
    }

    // Check whether specifying material via mesh
    bool zflag = std::all_of(mesh_regions.begin(), mesh_regions.end(),
            [](MeshRegion const &mr) {
                return mr.material != -1;
            });

    for (int i = 0; i < (int) mesh_regions.size(); i++) {
        MeshRegion &mr = mesh_regions[i];
        auto &sides = mr.sides;

        if (mr.type == MeshRegion::Type::UNKNOWN) {
            err.fail("ERROR: unrecognised region type");
            return;
        }

        if (mr.dims[0] <= 0) {
            err.fail("ERROR: region L dimension <= 0.0");
            return;
        }

        if (mr.dims[1] <= 0) {
            err.fail("ERROR: region K dimension <= 0.0");
            return;
        }

        if (mr.tol < 0.0) {
            err.fail("ERROR: region tol < 0.0");
            return;
        }

        if (mr.om < 0.0) {
            err.fail("ERROR: region om < 0.0");
            return;
        }

        if (zflag && (mr.material < 0 || mr.material >= sizes.nmat)) {
            err.fail("ERROR: region material index out of range");
            return;
        }

        for (int j = 0; j < (int) sides.size(); j++) {
            MeshRegion::Side &mrs = sides[j];
            auto &segments = mrs.segments;

            for (int k = 0; k < (int) segments.size(); k++) {
                MeshRegion::Side::Segment &mrss = segments[k];
                if (mrss.type == MeshRegion::Side::Segment::Type::UNKNOWN) {
                    err.fail("ERROR: unrecognised segment type");
                    return;
                }

                // Default boundary condition type is FREE so don't need to check
            }

            // If this is a multi-segment side...
            if (segments.size() > 1) {
                auto is_point = [](MeshRegion::Side::Segment const &mrss) {
                    return mrss.type == MeshRegion::Side::Segment::Type::POINT;
                };

                if (std::any_of(segments.begin(), segments.end(), is_point)) {
                    err.fail("ERROR: point as part of multi-segment side");
                    return;
                }

                // Check if the first two segments are connected by trying all
                // the permutations of their start and end points
                double l1, k1, l2, k2, l3, k3, l4, k4;

                l1 = segments[0].pos[2];
                k1 = segments[0].pos[3];
                l2 = segments[1].pos[0];
                k2 = segments[1].pos[1];

                if ((l1 != l2) || (k1 != k2)) {
                    l3 = segments[1].pos[2];
                    k3 = segments[1].pos[3];

                    if ((l1 != l3) || (k1 != k2)) {
                        l4 = segments[0].pos[0];
                        k4 = segments[0].pos[1];

                        if ((l4 != l2) || (k4 != k2)) {
                            if ((l4 != l3) || (k4 != k3)) {
                                err.fail("ERROR: segment failed to form "
                                         "continuous side");
                                return;
                            } else {
                                segments[0].pos[0] = l1;
                                segments[0].pos[1] = k1;
                                segments[0].pos[2] = l4;
                                segments[0].pos[3] = k4;
                                segments[1].pos[0] = l3;
                                segments[1].pos[1] = k3;
                                segments[1].pos[2] = l2;
                                segments[1].pos[3] = k2;
                            }
                        } else {
                            segments[0].pos[0] = l1;
                            segments[0].pos[1] = k1;
                            segments[0].pos[2] = l4;
                            segments[0].pos[3] = k4;
                        }
                    } else {
                        segments[1].pos[0] = l3;
                        segments[1].pos[1] = k3;
                        segments[1].pos[2] = l2;
                        segments[1].pos[3] = k2;
                    }
                }

                // If there's more than two segments in the side then use the
                // corrected first two segments to correct the rest.
                if (segments.size() > 2) {
                    for (int k = 1; k < (int) segments.size() - 1; k++) {
                        l1 = segments[k].pos[2];
                        k1 = segments[k].pos[3];
                        l2 = segments[k+1].pos[0];
                        k2 = segments[k+1].pos[1];

                        if ((l1 != l2) || (k1 != k2)) {
                            l3 = segments[k+1].pos[2];
                            k3 = segments[k+1].pos[3];

                            if ((l1 != l3) || (k1 != k3)) {
                                err.fail("ERROR: segment failed to form "
                                         "continuous side");
                                return;
                            } else {
                                segments[k+1].pos[0] = l3;
                                segments[k+1].pos[1] = k3;
                                segments[k+1].pos[2] = l2;
                                segments[k+1].pos[3] = k2;
                            }
                        }
                    }
                } // segments.size() > 2
            } // segments.size() > 1
        } // for each side

        // Check the sides form a closed polygon
        assert(sides.size() == 4);
        double l1, k1, l2, k2, l3, k3, l4, k4;
        int nsegs1, nsegs2;

        nsegs1 = sides[0].segments.size();
        l1 = sides[0].segments[nsegs1-1].pos[2];
        k1 = sides[0].segments[nsegs1-1].pos[3];
        l2 = sides[1].segments[0].pos[0];
        k2 = sides[1].segments[0].pos[1];

        if ((l1 != l2) || (k1 != k2)) {
            nsegs2 = sides[1].segments.size();
            l3 = sides[1].segments[nsegs2-1].pos[2];
            k3 = sides[1].segments[nsegs2-1].pos[3];

            if ((l1 != l3) || (k1 != k3)) {
                l4 = sides[0].segments[0].pos[0];
                k4 = sides[0].segments[0].pos[1];

                if ((l4 != l2) || (k4 != k2)) {
                    if ((l4 != l3) || (k4 != k3)) {
                        err.fail("ERROR: mesh region not closed");
                        return;
                    } else {
                        if (nsegs1 == 1) {
                            sides[0].segments[0].pos[0] = l1;
                            sides[0].segments[0].pos[1] = k1;
                            sides[0].segments[0].pos[2] = l4;
                            sides[0].segments[0].pos[3] = k4;
                        } else {
                            for (int l = 0; l < nsegs1; l++) {
                                std::swap(
                                    sides[0].segments[l].pos[0],
                                    sides[0].segments[l].pos[2]);
                                std::swap(
                                    sides[0].segments[l].pos[1],
                                    sides[0].segments[l].pos[3]);
                            }
                        }

                        if (nsegs2 == 1) {
                            sides[1].segments[0].pos[0] = l3;
                            sides[1].segments[0].pos[1] = k3;
                            sides[1].segments[0].pos[2] = l2;
                            sides[1].segments[0].pos[3] = k2;
                        } else {
                            for (int l = 0; l < nsegs2; l++) {
                                std::swap(
                                    sides[1].segments[l].pos[0],
                                    sides[1].segments[l].pos[2]);
                                std::swap(
                                    sides[1].segments[l].pos[1],
                                    sides[1].segments[l].pos[3]);
                            }
                        }
                    }
                } else {
                    if (nsegs1 == 1) {
                        sides[0].segments[0].pos[0] = l1;
                        sides[0].segments[0].pos[1] = k1;
                        sides[0].segments[0].pos[2] = l4;
                        sides[0].segments[0].pos[3] = k4;
                    } else {
                        for (int l = 0; l < nsegs1; l++) {
                            std::swap(
                                sides[0].segments[l].pos[0],
                                sides[0].segments[l].pos[2]);
                            std::swap(
                                sides[0].segments[l].pos[1],
                                sides[0].segments[l].pos[3]);
                        }
                    }
                }
            } else {
                if (nsegs2 == 1) {
                    sides[1].segments[0].pos[0] = l3;
                    sides[1].segments[0].pos[1] = k3;
                    sides[1].segments[0].pos[2] = l2;
                    sides[1].segments[0].pos[3] = k2;
                } else {
                    for (int l = 0; l < nsegs2; l++) {
                        std::swap(
                            sides[1].segments[l].pos[0],
                            sides[1].segments[l].pos[2]);
                        std::swap(
                            sides[1].segments[l].pos[1],
                            sides[1].segments[l].pos[3]);
                    }
                }
            }
        }

        // Correct the remaining sides
        for (int k = 1; k < 3; k++) {
            nsegs1 = sides[k].segments.size();
            l1 = sides[k].segments[nsegs1-1].pos[2];
            k1 = sides[k].segments[nsegs1-1].pos[3];
            l2 = sides[k+1].segments[0].pos[0];
            k2 = sides[k+1].segments[0].pos[1];

            if ((l1 != l2) || (k1 != k2)) {
                nsegs2 = sides[k+1].segments.size();
                l3 = sides[k+1].segments[nsegs2-1].pos[2];
                k3 = sides[k+1].segments[nsegs2-1].pos[3];

                if ((l1 != l3) || (k1 != k3)) {
                    err.fail("ERROR: region not closed");
                    return;
                } else {
                    if (nsegs2 == 1) {
                        sides[k+1].segments[0].pos[0] = l3;
                        sides[k+1].segments[0].pos[1] = k3;
                        sides[k+1].segments[0].pos[2] = l2;
                        sides[k+1].segments[0].pos[3] = k2;
                    } else {
                        for (int l = 0; l < nsegs2; l++) {
                            std::swap(
                                sides[k+1].segments[l].pos[0],
                                sides[k+1].segments[l].pos[2]);
                            std::swap(
                                sides[k+1].segments[l].pos[1],
                                sides[k+1].segments[l].pos[3]);
                        }
                    }
                }
            }
        }

        nsegs1 = sides[3].segments.size();
        l1 = sides[3].segments[nsegs1-1].pos[2];
        k1 = sides[3].segments[nsegs1-1].pos[3];
        l2 = sides[0].segments[0].pos[0];
        k2 = sides[0].segments[0].pos[1];

        if ((l1 != l2) || (k1 != k2)) {
            err.fail("ERROR: region not closed");
            return;
        }

    } // for each region
}



#ifdef BOOKLEAF_DEBUG
void
dumpMeshRegionData(std::string filename, MeshRegion const &mr)
{
    Error err;

#ifdef BOOKLEAF_MPI_SUPPORT
    // Serialise the dumps per-process if we're running under MPI
    int nproc, rank;
    TYPH_Get_Size(&nproc);
    TYPH_Get_Rank(&rank);

    filename += std::to_string(rank);

    TYPH_Barrier();
    for (int ip = 0; ip < nproc; ip++) {
        if (ip == rank) {
#endif

            std::ofstream of(filename.c_str());
            if (!of.is_open()) {
                FAIL_WITH_LINE(err, "ERROR: couldn't open file " + filename);
                std::exit(EXIT_FAILURE);
            }

            unsigned long long *ptr;
            std::bitset<64> bs;

            int const no_l = mr.dims[0] + 1;
            int const no_k = mr.dims[1] + 1;
            #define IXm(i, j) (index2D(i, j, no_l))

            // Dims
            of << "dims\n";
            of << 2 << "\n";
            of << "integer" << "\n";
            of << mr.dims[0] << "\n";
            of << mr.dims[1] << "\n";
            of << "\n";

            // Type
            of << "type\n";
            of << 1 << "\n";
            of << "integer" << "\n";
            of << (int) mr.type << "\n";
            of << "\n";

            // Material
            of << "material\n";
            of << 1 << "\n";
            of << "integer" << "\n";
            of << mr.material << "\n";
            of << "\n";

            // Convergence/tolerance
            of << "tol_om\n";
            of << 2 << "\n";
            of << "double" << "\n";
            ptr = (unsigned long long *) &mr.tol;
            bs = std::bitset<64>(*ptr);
            of << bs << "\n";
            ptr = (unsigned long long *) &mr.om;
            bs = std::bitset<64>(*ptr);
            of << bs << "\n";
            of << "\n";

            // # iterations
            of << "no_it\n";
            of << 1 << "\n";
            of << "integer" << "\n";
            of << mr.no_it << "\n";
            of << "\n";

            // rr
            of << "rr\n";
            of << no_l * no_k << "\n";
            of << "double" << "\n";
            for (int kk = 0; kk < no_k; kk++) {
                for (int ll = 0; ll < no_l; ll++) {
                    ptr = (unsigned long long *) &mr.rr[IXm(ll, kk)];
                    bs = std::bitset<64>(*ptr);
                    of << bs << "\n";
                }
            }
            of << "\n";

            // ss
            of << "ss\n";
            of << no_l * no_k << "\n";
            of << "double" << "\n";
            for (int kk = 0; kk < no_k; kk++) {
                for (int ll = 0; ll < no_l; ll++) {
                    ptr = (unsigned long long *) &mr.ss[IXm(ll, kk)];
                    bs = std::bitset<64>(*ptr);
                    of << bs << "\n";
                }
            }
            of << "\n";

            // bc
            of << "bc\n";
            of << no_l * no_k << "\n";
            of << "integer" << "\n";
            for (int kk = 0; kk < no_k; kk++) {
                for (int ll = 0; ll < no_l; ll++) {
                    of << mr.bc[IXm(ll, kk)] << "\n";
                }
            }
            of << "\n";

            // merged
            of << "merged\n";
            of << no_l * no_k << "\n";
            of << "integer" << "\n";
            for (int kk = 0; kk < no_k; kk++) {
                for (int ll = 0; ll < no_l; ll++) {
                    of << (int) mr.merged[IXm(ll, kk)] << "\n";
                }
            }
            of << "\n";

            // r_wgt
            of << "r_wgt\n";
            of << 8 << "\n";
            of << "double" << "\n";
            for (int i = 0; i < 8; i++) {
                ptr = (unsigned long long *) &mr.r_wgt[i];
                bs = std::bitset<64>(*ptr);
                of << bs << "\n";
            }
            of << "\n";

            // s_wgt
            of << "s_wgt\n";
            of << 8 << "\n";
            of << "double" << "\n";
            for (int i = 0; i < 8; i++) {
                ptr = (unsigned long long *) &mr.s_wgt[i];
                bs = std::bitset<64>(*ptr);
                of << bs << "\n";
            }
            of << "\n";

            // wgt
            of << "wgt\n";
            of << 16 << "\n";
            of << "double" << "\n";
            for (int i = 0; i < 16; i++) {
                ptr = (unsigned long long *) &mr.wgt[i];
                bs = std::bitset<64>(*ptr);
                of << bs << "\n";
            }
            of << "\n";

            of.close();
            if (err.failed()) std::exit(EXIT_FAILURE);

#ifdef BOOKLEAF_MPI_SUPPORT
        }
        TYPH_Barrier();
    }
#endif
}
#endif // BOOKLEAF_DEBUG

} // namespace setup
} // namespace bookleaf
