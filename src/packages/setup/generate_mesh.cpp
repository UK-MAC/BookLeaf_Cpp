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
#include "packages/setup/generate_mesh.h"

#include <cstring>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>

#include "common/error.h"
#include "common/sizes.h"
#include "common/cmd_args.h"
#include "utilities/data/global_configuration.h"
#include "common/data_control.h"
#include "common/timer_control.h"
#include "packages/setup/mesh_region.h"



namespace bookleaf {
namespace setup {
namespace {

#define IX(i, j) (index2D((i-1), (j-1), no_l))

void
initialiseRegionWeights(MeshRegion &mr, double zerocut, Error &err)
{
    // Set weights
    memset(mr.r_wgt, 0, sizeof(double) * 8);
    memset(mr.s_wgt, 0, sizeof(double) * 8);

    switch (mr.type) {
    case MeshRegion::Type::LIN1:
        mr.r_wgt[1] = 1.0;
        mr.r_wgt[6] = 1.0;
        mr.s_wgt[1] = 1.0;
        mr.s_wgt[6] = 1.0;
        break;

    case MeshRegion::Type::LIN2:
        mr.r_wgt[3] = 1.0;
        mr.r_wgt[4] = 1.0;
        mr.s_wgt[3] = 1.0;
        mr.s_wgt[4] = 1.0;
        break;

    case MeshRegion::Type::USER:
        {
            std::copy(mr.wgt, mr.wgt+8, mr.r_wgt);
            bool nz = std::any_of(mr.wgt+8, mr.wgt+16,
                    [](double v) { return v != 0.; });
            if (nz) {
                double sum = std::accumulate(mr.r_wgt, mr.r_wgt+8, 0,
                        [](double sum, double v) { return sum += std::abs(v); });
                if (std::abs(sum) > zerocut) {
                    std::copy(mr.wgt+8, mr.wgt+16, mr.s_wgt);
                }

            } else {
                std::copy(mr.r_wgt, mr.r_wgt+8, mr.s_wgt);
            }
        }
        break;

    default:
        // Do nothing
        break;
    }

    double tl0 = std::accumulate(mr.r_wgt, mr.r_wgt+8, 0.);
    double tl1 = std::accumulate(mr.s_wgt, mr.s_wgt+8, 0.);
    if (std::abs(tl0) < zerocut || std::abs(tl1) < zerocut) {
        err.fail("ERROR: ill defined user weights in mesh generation");
    }

    for (int i = 0; i < 8; i++) {
        mr.r_wgt[i] /= tl0;
        mr.s_wgt[i] /= tl1;
    }
}



void
calculateLineBoundary(MeshRegion &mr, int side, int segment, int l1, int l2,
        int k1, int k2, double zerocut, Error &err)
{
    MeshRegion::Side::Segment &mrss = mr.sides[side].segments[segment];

    int const no_l = mr.dims[0] + 1;
    int const no_k __attribute__((unused)) = mr.dims[1] + 1;

    double s1 = mrss.pos[0];
    double r1 = mrss.pos[1];
    double s2 = mrss.pos[2];
    double r2 = mrss.pos[3];
    double w1 = mrss.pos[4];
    double w2 = mrss.pos[5];

    int kmin = std::min(k1, k2);
    int kmax = std::max(k1, k2);
    int lmin = std::min(l1, l2);
    int lmax = std::max(l1, l2);

    double dr = r2 - r1;
    double ds = s2 - s1;

    int il = std::max(1, lmax - lmin);
    int ik = std::max(1, kmax - kmin);

    double fac = w1 + w2;
    bool weighted = false;
    if (fac > zerocut) {
        weighted = true;
        w1 /= fac;
        w2 /= fac;
        if (mr.type == MeshRegion::Type::USER) {
            err.fail("");
            return;
        }
    }

    double const tol = mr.tol;
    double const om = mr.om;

    if (lmin == lmax) {
        double dd = (double) ik;
        for (int kk = kmin; kk <= kmax; kk++) {
            int ki = kk;
            if (kmin == k2) ki = kmax + kmin - kk;
            mr.rr[IX(l1, ki)] = r1 + (double) (kk - kmin) * dr / dd;
            mr.ss[IX(l1, ki)] = s1 + (double) (kk - kmin) * ds / dd;
            mr.bc[IX(l1, ki)] = (int) mrss.bc;
        }

        if (weighted) {
            int k3 = kmin + 1;
            int k4 = kmax - 1;
            while (true) {
                bool exit_status = true;
                for (int kk = k3; kk <= k4; kk++) {
                    int km = kk - 1;
                    int kp = kk + 1;

                    dr = w1 * mr.rr[IX(l1, km)] +
                         w2 * mr.rr[IX(l1, kp)] - mr.rr[IX(l1, kk)];
                    ds = w1 * mr.ss[IX(l1, km)] +
                         w2 * mr.ss[IX(l1, kp)] - mr.ss[IX(l1, kk)];

                    if (std::abs(dr) > tol || std::abs(ds) > tol) {
                        exit_status = false;
                    }

                    mr.rr[IX(l1, kk)] += om * dr;
                    mr.ss[IX(l1, kk)] += om * ds;
                }
                if (exit_status) break;
            }
        }

    } else {
        double dd = (double) il;
        for (int ll = lmin; ll <= lmax; ll++) {
            int li = ll;
            if (lmin == l2) li = lmax + lmin - ll;
            mr.rr[IX(li, k1)] = r1 + (double) (ll - lmin) * dr / dd;
            mr.ss[IX(li, k1)] = s1 + (double) (ll - lmin) * ds / dd;
            mr.bc[IX(li, k1)] = (int) mrss.bc;
        }

        if (weighted) {
            int l3 = lmin + 1;
            int l4 = lmax - 1;
            while (true) {
                bool exit_status = true;
                for (int ll = l3; ll <= l4; ll++) {
                    int lm = ll - 1;
                    int lp = ll + 1;

                    dr = w1 * mr.rr[IX(lm, k1)] +
                         w2 * mr.rr[IX(lp, k1)] - mr.rr[IX(ll, k1)];
                    ds = w1 * mr.ss[IX(lm, k1)] +
                         w2 * mr.ss[IX(lp, k1)] - mr.ss[IX(ll, k1)];

                    if (std::abs(dr) > tol || std::abs(ds) > tol) {
                        exit_status = false;
                    }

                    mr.rr[IX(ll, k1)] += om * dr;
                    mr.ss[IX(ll, k1)] += om * ds;
                }
                if (exit_status) break;
            }
        }
    }
}



void
calculateArcBoundary(MeshRegion &mr, int side, int segment, int l1, int l2,
        int k1, int k2, double zerocut, Error &err __attribute__((unused)))
{
    MeshRegion::Side::Segment &mrss = mr.sides[side].segments[segment];

    int const no_l = mr.dims[0] + 1;
    int const no_k __attribute__((unused)) = mr.dims[1] + 1;

    //! Argument list
    //INTEGER(KIND=ink),INTENT(IN)    :: ireg,i_seg,ind,l1,l2,k1,k2
    //REAL(KIND=rlk),   INTENT(IN)    :: zerocut
    //! Local
    //INTEGER(KIND=ink)        :: lmin,lmax,ll,il,li,l3,l4,lm,lp
    //INTEGER(KIND=ink)        :: kmin,kmax,kk,ik,ki,k3,k4,km,kp
    //REAL(KIND=rlk)           :: r0,r1,r2,s0,s1,s2,fac,d1,d2,theta,theta1
    //REAL(KIND=rlk)           :: theta2,dtheta,w1,w2,w3,w4,dd,dl,tol,om
    //LOGICAL(KIND=lok)        :: fl,weighted,exit_status
    //REAL(KIND=rlk),PARAMETER :: PI=3.1415926535897932385_rlk,                  &
//&                               TWOPI=6.2831853071795864770_rlk

    bool fl = false;
    double s1, r1, s2, r2;
    if (mrss.type == MeshRegion::Side::Segment::Type::ARC_C) {
        s1 = mrss.pos[2];
        r1 = mrss.pos[3];
        s2 = mrss.pos[0];
        r2 = mrss.pos[1];
        fl = true;
    } else {
        s1 = mrss.pos[0];
        r1 = mrss.pos[1];
        s2 = mrss.pos[2];
        r2 = mrss.pos[3];
        fl = false;
    }

    double r0 = mrss.pos[4];
    double s0 = mrss.pos[5];
    double w1 = mrss.pos[6];
    double w2 = mrss.pos[7];

    int kmin = std::min(k1, k2);
    int kmax = std::max(k1, k2);
    int lmin = std::min(l1, l2);
    int lmax = std::max(l1, l2);

    double fac = w1 + w2;
    bool weighted =false;
    if (fac > zerocut) {
        w1 /= fac;
        w2 /= fac;
        weighted = true;
    }

    double dd = s1 - s0;
    double w3 = r1 - r0;
    double d2;
    if (std::abs(w3) < zerocut) {
        d2 = zerocut;
    } else {
        d2 = w3;
    }

    double theta1 = std::atan(dd / d2);
    if (d2 < 0.) {
        theta1 += M_PI;
    } else if (dd < 0.) {
        theta1 += 2.*M_PI;
    }

    double d1 = s2 - s0;
    double w4 = r2 - r0;
    if (std::abs(w4) < zerocut) {
        d2 = zerocut;
    } else {
        d2 = w4;
    }
    double theta2 = std::atan(d1 / d2);
    if (d2 < 0.) {
        theta2 += M_PI;
    } else if (d1 < 0.) {
        theta2 += 2.*M_PI;
    }

    if (theta2 > theta1) theta2 -= 2.*M_PI;
    d2 = std::sqrt(w4*w4 + d1*d1);
    d1 = std::sqrt(w3*w3 + dd*dd);
    double tol = mr.tol;
    //double om = mr.om;

    if (lmin == lmax) {
        int ik = kmax - kmin;
        dd = (double) ik;
        double dtheta = theta1 - theta2;
        for (int kk = kmin; kk <= kmax; kk++) {
            int ki = kk;
            if (kmin == k2) ki = kmin + kmax - kk;
            double theta = theta1 - (double) (kk - kmin) * dtheta / dd;
            w3 = -1. / dd;
            double dl = (d1 - d2) / dtheta;
            dl = d1 + dl*(theta - theta1);
            mr.rr[IX(l1, ki)] = r0 + dl * std::cos(theta);
            mr.ss[IX(l1, ki)] = s0 + dl * std::sin(theta);
            mr.bc[IX(l1, ki)] = (int) mrss.bc;
            if (mr.ss[IX(l1, ki)] < 0.) {
                w3 = theta - w3 + std::asin(-s0 / dl);
                theta = std::asin(-s0 / dl);
            }
        }
        mr.rr[IX(l1, k1)] = r1;
        mr.rr[IX(l2, k2)] = r2;
        mr.ss[IX(l1, k1)] = s1;
        mr.ss[IX(l2, k2)] = s2;
        if (weighted) {
            int k3 = kmin + 1;
            int k4 = kmax - 1;
            for (int kk = kmin; kk <= kmax; kk++) {
              int ki = kk;
              if (kmin == k2) ki = kmax + kmin - kk;
              mr.rr[IX(l1, ki)] = theta1 - (double) (kk - kmin) * dtheta / dd;
            }
            while (true) {
                bool exit_status = true;
                for (int kk = k3; kk <= k4; kk++) {
                    int km = kk - 1;
                    int kp = kk + 1;
                    d1 = w1 * mr.rr[IX(l1, km)] + w2 * mr.rr[IX(l1, kp)] -
                        mr.rr[IX(l1, kk)];
                    if (std::abs(d1) > tol) exit_status = false;
                    mr.rr[IX(l1, kk)] += d1;
                }
                if (exit_status) break;
            }
            for (int kk = k3; kk <= k4; kk++) {
                double theta = mr.rr[IX(l1, kk)];
                double dl = (d1 - d2) / dtheta;
                dl = d1 + dl*(theta - theta1);
                mr.rr[IX(l1, kk)] = r0 + dl*std::cos(theta);
                mr.ss[IX(l1, kk)] = r0 + dl*std::sin(theta);
            }
        }
    } else {
        int il = lmax - lmin;
        dd = (double) il;
        double dtheta = theta1 - theta2;
        for (int ll = lmin; ll <= lmax; ll++) {
            int li = ll;
            if (lmin == l2) li = lmax + lmin - ll;
            double theta = theta1 - (double) (ll - lmin) * dtheta / dd;
            w3 = -1. / dd;
            double dl = (d1 - d2) / dtheta;
            dl = d1 + dl*(theta - theta1);
            mr.rr[IX(li, k1)] = r0 + dl * std::cos(theta);
            mr.ss[IX(li, k1)] = s0 + dl * std::sin(theta);
            mr.bc[IX(li, k1)] = (int) mrss.bc;
            if (mr.ss[IX(li, k1)] < 0.) {
                w3 = theta - w3 + std::asin(-s0 / dl);
                theta = std::asin(-s0 / dl);
            }
        }
        mr.rr[IX(l1, k1)] = r1;
        mr.rr[IX(l2, k2)] = r2;
        mr.ss[IX(l1, k1)] = s1;
        mr.ss[IX(l2, k2)] = s2;
        if (weighted) {
            int l3 = lmin + 1;
            int l4 = lmax - 1;
            for (int ll = lmin; ll <= lmax; ll++) {
                int li = ll;
                if (lmin == l2) li = lmax + lmin - ll;
                mr.rr[IX(li, k1)] = theta1 - (double) (ll - lmin) * dtheta / dd;
            }
            while (true) {
                bool exit_status = true;
                for (int ll = l3; ll <= l4; ll++) {
                    int lm = ll - 1;
                    int lp = ll + 1;
                    d1 = w1 * mr.rr[IX(lm, k1)] + w2 * mr.rr[IX(lp, k1)] -
                        mr.rr[IX(ll, k1)];
                    if (std::abs(d1) > tol) exit_status = false;
                    mr.rr[IX(ll, k1)] += d1;
                }
                if (exit_status) break;
            }
            for (int ll = l3; ll <= l4; ll++) {
                double theta = mr.rr[IX(ll, k1)];
                double dl = (d1 - d2) / dtheta;
                dl = d1 + dl*(theta - theta1);
                mr.rr[IX(ll, k1)] = r0 + dl*std::cos(theta);
                mr.ss[IX(ll, k1)] = r0 + dl*std::sin(theta);
            }
        }
    }

    if (fl) {
        if (l1 != l2) {
            for (int ll = 0; ll <= (lmax-lmin)/2; ll++) {
                int li = lmin + ll;
                int il = lmax - ll;
                w3 = mr.rr[IX(li, k1)];
                w4 = mr.ss[IX(li, k1)];
                mr.rr[IX(li, k1)] = mr.rr[IX(il, k1)];
                mr.ss[IX(li, k1)] = mr.ss[IX(il, k1)];
                mr.rr[IX(il, k1)] = w3;
                mr.ss[IX(il, k1)] = w4;
            }
        } else {
            for (int kk = 0; kk <= (kmax-kmin)/2; kk++) {
                int ki = kmin + kk;
                int ik = kmax - kk;
                w3 = mr.rr[IX(l1, ki)];
                w4 = mr.ss[IX(l1, ki)];
                mr.rr[IX(l1, ki)] = mr.rr[IX(l1, ik)];
                mr.ss[IX(l1, ki)] = mr.ss[IX(l1, ik)];
                mr.rr[IX(l1, ik)] = w3;
                mr.ss[IX(l1, ik)] = w4;
            }
        }
    }
}



void
calculateRegionBoundary(MeshRegion &mr, double zerocut, Error &err)
{
    int const no_l = mr.dims[0] + 1;
    int const no_k = mr.dims[1] + 1;

    for (int i = 0; i < 4; i++) {
        MeshRegion::Side &mrs = mr.sides[i];

        int l1, l2, k1, k2;
        l1 = l2 = k1 = k2 = 0;
        switch (i) {
        case 0:
            l1 = k1 = l2 = 1;
            k2 = no_k;
            break;

        case 1:
            l1 = 1;
            k1 = no_k;
            l2 = no_l;
            k2 = no_k;
            break;

        case 2:
            l1 = no_l;
            k1 = no_k;
            l2 = no_l;
            k2 = 1;
            break;

        case 3:
            l1 = no_l;
            k1 = l2 = k2 = 1;
            break;

        default:
            assert(false);
        }

        int const no_seg = mrs.segments.size();
        if (no_seg == 1) {
            switch (mrs.segments[0].type) {
            case MeshRegion::Side::Segment::Type::LINE:
            case MeshRegion::Side::Segment::Type::LINK:
                calculateLineBoundary(mr, i, 0, l1, l2, k1, k2, zerocut, err);
                if (err.failed()) {
                    err.fail("ERROR: consistency between user weights and "
                            "line not yet coded");
                    return;
                }
                break;

            case MeshRegion::Side::Segment::Type::ARC_C:
            case MeshRegion::Side::Segment::Type::ARC_A:
                calculateArcBoundary(mr, i, 0, l1, l2, k1, k2, zerocut, err);
                break;

            case MeshRegion::Side::Segment::Type::POINT:
                err.fail("ERROR: side_type is POINT but DJN off");
                return;

            default:
                err.fail("ERROR: unexpected UNKNOWN segment");
                return;
          }

        } else {
            // Other options
            assert(false && "not yet implemented");
        }
    }
}



void
calculateRegionInterior(MeshRegion &mr, double zerocut,
        Error &err __attribute__((unused)))
{
    int const no_l = mr.dims[0] + 1;
    int const no_k = mr.dims[1] + 1;

    // Set equipotential flag
    bool ep = (mr.type == MeshRegion::Type::EQUI);

    // Calculate initial guess at interior points
    int l1, k1, l2, k2;
    l1 = 2;
    k1 = 2;
    l2 = no_l - 1;
    k2 = no_k - 1;

    double tl0, tl1;
    tl0 = 0.25;
    tl1 = 0.25;
    double tk0 = 0.25;
    double tk1 = 0.25;
    double fac = 1.0;

    if (!ep) {
        tl0 = mr.r_wgt[6];
        tl1 = mr.r_wgt[1];
        tk0 = mr.r_wgt[3];
        tk1 = mr.r_wgt[4];
        fac = tl0 + tl1 + tk0 + tk1;
        if (std::abs(fac) <= zerocut) fac = 1.0;
    }

    tl0 = 2. * tl0 / fac;
    tl1 = 2. * tl1 / fac;
    tk0 = 2. * tk0 / fac;
    tk1 = 2. * tk1 / fac;

    double ww, wl0, wl1, wk0, wk1;
    for (int kk = k1; kk <= k2; kk++) {
        for (int ll = l1; ll <= l2; ll++) {
            ww  = (double) (l2 - l1) + 2.;
            wl0 = tl0 * ((double) (l2 - ll) + 1.) / ww;
            wl1 = tl1 * ((double) (ll - l1) + 1.) / ww;
            ww  = (double) (k2 - k1) + 2.;
            wk0 = tk0 * ((double) (k2 - kk) + 1.) / ww;
            wk1 = tk1 * ((double) (kk - k1) + 1.) / ww;
            mr.rr[IX(ll, kk)] = wk0 * mr.rr[IX(ll, k1-1)] +
                                wk1 * mr.rr[IX(ll, k2+1)] +
                                wl0 * mr.rr[IX(l1-1, kk)] +
                                wl1 * mr.rr[IX(l2+1, kk)];
            mr.ss[IX(ll, kk)] = wk0 * mr.ss[IX(ll, k1-1)] +
                                wk1 * mr.ss[IX(ll, k2+1)] +
                                wl0 * mr.ss[IX(l1-1, kk)] +
                                wl1 * mr.ss[IX(l2+1, kk)];
        }
    }

    // Calculate interior points
    int no_it = 0;
    double const tol = mr.tol;
    double const om = mr.om;
    int const max_no_it = 8 * no_l * no_k;

    double r1, r2, r3, r4, r5, r6, r7, r8;
    double s1, s2, s3, s4, s5, s6, s7, s8;
    if (ep) {
        // To ensure r1-8,s1-8 initialised, not strictly necessary but keeps
        // gcc based compilers happy.
        r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = 0.;
        s1 = s2 = s3 = s4 = s5 = s6 = s7 = s8 = 0.;
    } else {
        r1 = mr.r_wgt[0];
        r2 = mr.r_wgt[1];
        r3 = mr.r_wgt[2];
        r4 = mr.r_wgt[3];
        r5 = mr.r_wgt[4];
        r6 = mr.r_wgt[5];
        r7 = mr.r_wgt[6];
        r8 = mr.r_wgt[7];
        s1 = mr.s_wgt[0];
        s2 = mr.s_wgt[1];
        s3 = mr.s_wgt[2];
        s4 = mr.s_wgt[3];
        s5 = mr.s_wgt[4];
        s6 = mr.s_wgt[5];
        s7 = mr.s_wgt[6];
        s8 = mr.s_wgt[7];
    }

    // Mesh loop
    while (true) {
        no_it = no_it + 1;
        bool exit_status = true;

        for (int kk = k1; kk <= k2; kk++) {
            for (int ll = l1; ll <= l2; ll++) {
                int lm = ll - 1;
                int lp = ll + 1;
                int km = kk - 1;
                int kp = kk + 1;
                if (ep) {
                    double r_psi = 0.5 * (mr.rr[IX(ll, kp)] - mr.rr[IX(ll, km)]);
                    double s_psi = 0.5 * (mr.ss[IX(ll, kp)] - mr.ss[IX(ll, km)]);
                    double r_phi = 0.5 * (mr.rr[IX(lp, kk)] - mr.rr[IX(lm, kk)]);
                    double s_phi = 0.5 * (mr.ss[IX(lp, kk)] - mr.ss[IX(lm, kk)]);
                    double zeta = s_psi*r_phi - s_phi*r_psi;
                    if (std::abs(zeta) < zerocut) continue;

                    double alph = s_psi*s_psi + r_psi*r_psi;
                    double beta = s_phi*s_psi + r_phi*r_psi;
                    double gamm = s_phi*s_phi + r_phi*r_phi;
                    r1 = r8 = 0.5 * beta;
                    r2 = r7 = alph;
                    r3 = r6 = -r1;
                    r4 = r5 = gamm;
                    fac = r1+r2+r3+r4+r5+r6+r7+r8;
                    if (fac < 1.0e-24) {
                        fac *= 1.0e14;
                        r1 *= 1.0e14;
                        r2 *= 1.0e14;
                        r3 *= 1.0e14;
                        r4 *= 1.0e14;
                        r5 *= 1.0e14;
                        r6 *= 1.0e14;
                        r7 *= 1.0e14;
                        r8 *= 1.0e14;
                    }

                    r1 /= fac;
                    r2 /= fac;
                    r3 /= fac;
                    r4 /= fac;
                    r5 /= fac;
                    r6 /= fac;
                    r7 /= fac;
                    r8 /= fac;

                    s1 = r1;
                    s2 = r2;
                    s3 = r3;
                    s4 = r4;
                    s5 = r5;
                    s6 = r6;
                    s7 = r7;
                    s8 = r8;
                } // if ep

                double dr = r1 * mr.rr[IX(lp, km)] +
                            r2 * mr.rr[IX(lp, kk)] +
                            r3 * mr.rr[IX(lp, kp)] +
                            r4 * mr.rr[IX(ll, km)] +
                            r5 * mr.rr[IX(ll, kp)] +
                            r6 * mr.rr[IX(lm, km)] +
                            r7 * mr.rr[IX(lm, kk)] +
                            r8 * mr.rr[IX(lm, kp)] - mr.rr[IX(ll, kk)];
                double ds = s1 * mr.ss[IX(lp, km)] +
                            s2 * mr.ss[IX(lp, kk)] +
                            s3 * mr.ss[IX(lp, kp)] +
                            s4 * mr.ss[IX(ll, km)] +
                            s5 * mr.ss[IX(ll, kp)] +
                            s6 * mr.ss[IX(lm, km)] +
                            s7 * mr.ss[IX(lm, kk)] +
                            s8 * mr.ss[IX(lm, kp)] - mr.ss[IX(ll, kk)];

                // If hasn't converged
                if (std::abs(dr) > tol || std::abs(ds) > tol) {
                    exit_status = false;
                    mr.rr[IX(ll, kk)] = mr.rr[IX(ll, kk)] + om*dr;
                    mr.ss[IX(ll, kk)] = mr.ss[IX(ll, kk)] + om*ds;
                }
            } // for ll
        } // for kk

        if (no_it > max_no_it) {
            std::cerr << "WARNING: maximum number of iterations exceeded "
                " whilst generating mesh for region";
            break;
        }

        if (exit_status) {
            mr.no_it = no_it;
            break;
        }
    } // mesh loop
}



void
calculateSizes(std::vector<MeshRegion> &mesh_regions, int &nel, int &nnd,
        Error &err)
{
    nel = nnd = 0;

    for (int ireg = 0; ireg < (int) mesh_regions.size(); ireg++) {
        MeshRegion &mr = mesh_regions[ireg];

        int const no_l = mr.dims[0] + 1;
        int const no_k __attribute__((unused)) = mr.dims[1] + 1;

        nel += mr.dims[0] * mr.dims[1];

        auto check_corner = [&](int ix1, int ix2, int ix3, int ix4) {
            if (((mr.bc[IX(ix1, ix2)] == 1) && (mr.bc[IX(ix3, ix4)] == 2)) ||
                ((mr.bc[IX(ix1, ix2)] == 2) && (mr.bc[IX(ix3, ix4)] == 1))) {
                mr.bc[IX(ix3, ix2)] = 3;
            }
            if (((mr.bc[IX(ix1, ix2)] == 6) && (mr.bc[IX(ix3, ix4)] == 1)) ||
                ((mr.bc[IX(ix1, ix2)] == 1) && (mr.bc[IX(ix3, ix4)] == 6))) {
                mr.bc[IX(ix3, ix2)] = 4;
            }
            if (((mr.bc[IX(ix1, ix2)] == 6) && (mr.bc[IX(ix3, ix4)] == 2)) ||
                ((mr.bc[IX(ix1, ix2)] == 2) && (mr.bc[IX(ix3, ix4)] == 6))) {
                mr.bc[IX(ix3, ix2)] = 5;
            }
            if (((mr.bc[IX(ix1, ix2)] == 8) && (mr.bc[IX(ix3, ix4)] == 1)) ||
                ((mr.bc[IX(ix1, ix2)] == 1) && (mr.bc[IX(ix3, ix4)] == 8))) {
                mr.bc[IX(ix3, ix2)] = 1;
            }
            if (((mr.bc[IX(ix1, ix2)] == 8) && (mr.bc[IX(ix3, ix4)] == 2)) ||
                ((mr.bc[IX(ix1, ix2)] == 2) && (mr.bc[IX(ix3, ix4)] == 8))) {
                mr.bc[IX(ix3, ix2)] = 2;
            }
            if (((mr.bc[IX(ix1, ix2)] == 8) && (mr.bc[IX(ix3, ix4)] == 3)) ||
                ((mr.bc[IX(ix1, ix2)] == 3) && (mr.bc[IX(ix3, ix4)] == 8))) {
                mr.bc[IX(ix3, ix2)] = 3;
            }
            if (((mr.bc[IX(ix1, ix2)] == 6) && (mr.bc[IX(ix3, ix4)] == 3)) ||
                ((mr.bc[IX(ix1, ix2)] == 3) && (mr.bc[IX(ix3, ix4)] == 6))) {
                err.fail("ERROR: inconsistent BC at corner");
                return;
            }
        };

        // Set corner BC
        if (!mr.merged[IX(1, 1)]) { check_corner(2, 1, 1, 2); }

        int i1 = mr.dims[1];
        int i2 = i1 + 1;
        if (!mr.merged[IX(1, i2)]) { check_corner(2, i2, 1, i1); }

        i1 = mr.dims[0];
        i2 = i1 + 1;
        if (!mr.merged[IX(i2, 1)]) { check_corner(i1, 1, i2, 2); }

        i1 = mr.dims[0] + 1;
        i2 = mr.dims[1] + 1;
        if (!mr.merged[IX(i1, i2)]) { check_corner(i1-1, i2, i1, i2-1); }

        for (int ll = 1; ll <= mr.dims[0]+1; ll++) {
            for (int kk = 1; kk <= mr.dims[1]+1; kk++) {
                if (!mr.merged[IX(ll, kk)]) nnd++;
            }
        }

    } // for ireg

    assert(nel > 0);
    assert(nnd > 0);
}



void
generateMeshRegions(std::vector<MeshRegion> &mesh_regions, int &nel,
        int &nnd, double const zerocut, Error &err)
{
    assert(mesh_regions.size() == 1);
    for (int ireg = 0; ireg < (int) mesh_regions.size(); ireg++) {
        MeshRegion &mr = mesh_regions[ireg];

        // Allocate
        int const no_l = mr.dims[0] + 1;
        int const no_k = mr.dims[1] + 1;

        mr.rr = new double[no_l * no_k];
        mr.ss = new double[no_l * no_k];
        mr.bc = new int[no_l * no_k];
        mr.merged = new unsigned char[no_l * no_k];

        memset(mr.rr, 0, sizeof(double) * no_l * no_k);
        memset(mr.ss, 0, sizeof(double) * no_l * no_k);
        memset(mr.bc, 0, sizeof(int) * no_l * no_k);
        memset(mr.merged, 0, sizeof(unsigned char) * no_l * no_k);

        initialiseRegionWeights(mr, zerocut, err);
        if (err.failed()) return;

        calculateRegionBoundary(mr, zerocut, err);
        if (err.failed()) return;

        calculateRegionInterior(mr, zerocut, err);
        if (err.failed()) return;
    }

    calculateSizes(mesh_regions, nel, nnd, err);
    if (err.failed()) return;
}

#undef IX

} // namespace



void
generateMesh(std::vector<MeshRegion> &mesh_regions, GlobalConfiguration &global,
        Sizes &sizes, TimerControl &timers, Error &err)
{
    ScopedTimer st(timers, TimerID::MESHGEN);

    // Check user input
    rationaliseMeshRegions(mesh_regions, sizes, err);
    if (err.failed()) return;

    // Generate mesh regions
    generateMeshRegions(mesh_regions, sizes.nel, sizes.nnd, global.zerocut, err);
    if (err.failed()) return;

#ifdef BOOKLEAF_DEBUG
    if (CMD_ARGS.dump_mesh) dumpMeshRegionData("mesh.dat", mesh_regions[0]);
#endif
}

} // namespace setup
} // namespace bookleaf
