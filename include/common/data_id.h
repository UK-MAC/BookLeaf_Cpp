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
#ifndef BOOKLEAF_COMMON_DATA_ID_H
#define BOOKLEAF_COMMON_DATA_ID_H



namespace bookleaf {

enum class DataID : int {
    ELDENSITY = 0,  // Element density
    ELENERGY,       // Element energy
    ELPRESSURE,     // Element pressure
    ELCS2,          // Element speed of sound
    ELVOLUME,       // Element volume
    A1,             //
    A2,             //
    A3,             //
    B1,             //
    B2,             //
    B3,             //
    CNWT,           //
    CNX,            // Element-corner X-position
    CNY,            // Element-corner Y-position
    CNVISCX,        //
    CNVISCY,        //
    ELVISC,         // Element viscosity
    ELMASS,         // Element mass
    CNMASS,         // Element-corner mass
    NDX,            // Mesh node X-coordinates
    NDY,            // Mesh node Y-coordinates
    NDU,            // Node u-velocity
    NDV,            // Node v-velocity
    INDTYPE,        // Mesh node type index
    IELSORT1,       //
    IELND,          // Element-node mapping
    INDEL,          // Node-element mapping
    INDELN,         // Node-element mapping number per node
    INDELF,         // Node-element mapping offset per node
    IELEL,          // Element-element mapping
    IELFC,          // Element-face mapping
    IELMAT,         // Mesh element material index
    IELREG,         // Mesh element region index
    IELSORT2,       //
    SPMASS,         // Subzonal-pressure masses

    IELLOCGLOB,     // Element local-global mapping
    INDLOCGLOB,     // Node local-global mapping

    CPDENSITY,      // Component density
    CPENERGY,       // Component energy
    CPPRESSURE,     // Component pressure
    CPCS2,          // Component speed of sound
    CPVOLUME,       // Component volume
    FRVOLUME,       // Volume fraction in multi-material element
    CPMASS,         // Component mass
    FRMASS,         //
    CPVISCX,        //
    CPVISCY,        //
    CPVISC,         //
    CPA1,           //
    CPA3,           //
    CPB1,           //
    CPB3,           //
    ICPMAT,         // Component material index
    ICPNEXT,        // Component linked list---next index
    ICPPREV,        // Component linked list---previous index
    IMXEL,          // Mixed-material element index
    IMXFCP,         // Mixed-material element component list head
    IMXNCP,         // Mixed-material element component list length

    // XXX(timrlaw): Scratch
    RSCRATCH11,     // 1D real-valued scratch
    RSCRATCH12,     //         "
    RSCRATCH13,     //         "
    RSCRATCH14,     //         "
    RSCRATCH15,     //         "
    RSCRATCH16,     //         "
    RSCRATCH17,     //         "
    RSCRATCH18,     //         "
    RSCRATCH21,     // 2D real-valued scratch
    RSCRATCH22,     //         "
    RSCRATCH23,     //         "
    RSCRATCH24,     //         "
    RSCRATCH25,     //         "
    RSCRATCH26,     //         "
    RSCRATCH27,     //         "
    RSCRATCH28,     //         "
    ISCRATCH11,     // 1D integer-valued scratch
    ZSCRATCH11,     // 1D boolean-valued scratch
    ICPSCRATCH11,   // 1D integer-valued scratch
    ICPSCRATCH12,   //         "
    RCPSCRATCH11,   // 1D real-valued scratch
    RCPSCRATCH21,   // 2D real-valued scratch
    RCPSCRATCH22,   //         "
    RCPSCRATCH23,   //         "
    RCPSCRATCH24,   //         "

    // XXX(timrlaw): Leave here to provide a count of above definitions
    COUNT,

    // XXX(timrlaw): Below values provide aliases for convenience within certain
    //               kernels.
    // Setup aliases
    SETUP_CNX = RSCRATCH21,
    SETUP_CNY = RSCRATCH22,

    // DT step aliases
    TIME_SCRATCH = RSCRATCH11,
    TIME_ELLENGTH = RSCRATCH12,
    TIME_CNU = RSCRATCH21,
    TIME_CNV = RSCRATCH22,
    TIME_DU = RSCRATCH23,
    TIME_DV = RSCRATCH24,
    TIME_DX = RSCRATCH25,
    TIME_DY = RSCRATCH26,
    TIME_STORE = RSCRATCH27,

    // Lagrangian step aliases
    LAG_CNU = RSCRATCH21,
    LAG_CNV = RSCRATCH22,
    LAG_CNFX = RSCRATCH23,
    LAG_CNFY = RSCRATCH24,
    LAG_ELENERGY0 = RSCRATCH11,
    LAG_NDAREA = RSCRATCH12,
    LAG_NDMASS = RSCRATCH13,
    LAG_NDUBAR = RSCRATCH14,
    LAG_NDVBAR = RSCRATCH15,
    LAG_CPFX = RCPSCRATCH21,
    LAG_CPFY = RCPSCRATCH22,
    LAG_CPU = RSCRATCH23,
    LAG_CPV = RSCRATCH24,

    // ALE step aliases
    ALE_STORE1 = RSCRATCH11,
    ALE_STORE2 = RSCRATCH12,
    ALE_STORE3 = RSCRATCH13,
    ALE_STORE4 = RSCRATCH14,
    ALE_STORE5 = RSCRATCH15,
    ALE_STORE6 = RSCRATCH16,
    ALE_STORE7 = RSCRATCH17,
    ALE_STORE8 = RSCRATCH18,
    ALE_FCDV = RSCRATCH21,
    ALE_FCDM = RSCRATCH22,
    ALE_FLUX = RSCRATCH23,
    ALE_RWORK1 = RSCRATCH24,
    ALE_RWORK2 = RSCRATCH25,
    ALE_RWORK3 = RSCRATCH26,
    ALE_CNU = RSCRATCH27,
    ALE_CNV = RSCRATCH28,
    ALE_INDSTATUS = ISCRATCH11,
    ALE_ZACTIVE = ZSCRATCH11
};

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_DATA_ID_H
