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
#ifndef BOOKLEAF_COMMON_CUDA_UTILS_H
#define BOOKLEAF_COMMON_CUDA_UTILS_H



// Taken these max and min double values from a CUDA library
#define NPP_MAXABS_64F ( 1.7976931348623158e+308 )
#define NPP_MINABS_64F ( 2.2250738585072014e-308 )

#ifdef __CUDA_ARCH__

#define BL_MAX(a, b) max(a, b)
#define BL_MIN(a, b) min(a, b)
#define BL_ABS(a) abs(a)
#define BL_FABS(a) fabs(a)
#define BL_SQRT(a) sqrt(a)
#define BL_POW(a, b) pow(a, b)
#define BL_EXP(a) exp(a)
#define BL_SIGN(a, b) copysign(a, b)

#else

#define BL_MAX(a, b) std::max(a, b)
#define BL_MIN(a, b) std::min(a, b)
#define BL_ABS(a) std::abs(a)
#define BL_FABS(a) std::fabs(a)
#define BL_SQRT(a) std::sqrt(a)
#define BL_POW(a, b) std::pow(a, b)
#define BL_EXP(a) std::exp(a)
#define BL_SIGN(a, b) std::copysign(a, b)

#endif



#endif // BOOKLEAF_COMMON_CUDA_UTILS_H
