/*
 * @cond LICENSE
 * ######################################################################################
 * # LGPL License                                                                       #
 * #                                                                                    #
 * # This file is part of the LightJason                                                #
 * # Copyright (c) 2015-19, LightJason (info@lightjason.org)                            #
 * # This program is free software: you can redistribute it and/or modify               #
 * # it under the terms of the GNU Lesser General Public License as                     #
 * # published by the Free Software Foundation, either version 3 of the                 #
 * # License, or (at your option) any later version.                                    #
 * #                                                                                    #
 * # This program is distributed in the hope that it will be useful,                    #
 * # but WITHOUT ANY WARRANTY; without even the implied warranty of                     #
 * # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                      #
 * # GNU Lesser General Public License for more details.                                #
 * #                                                                                    #
 * # You should have received a copy of the GNU Lesser General Public License           #
 * # along with this program. If not, see http://www.gnu.org/licenses/                  #
 * ######################################################################################
 * @endcond
 */

// -----
// agent for testing blas actions
// @iteration 2
// @testcount 1
// -----

// initial-goal
!test.

/**
 * base test
 */
+!test <-
    !testblas
.


/**
 * test blas
 */
+!testblas <-
         M = .math/blas/matrix/create(2,2);
         .math/blas/matrix/set(0,0, 1, M);
         .math/blas/matrix/set(0,1, 2, M);
         .math/blas/matrix/set(1,0, 3, M);
         .math/blas/matrix/set(1,1, 4, M);

         Det = .math/blas/matrix/determinant(M);
         [EVal|EVec] = .math/blas/matrix/eigen(M);
         .test/print("blas matrix", M,Det,EVal,EVec);
         .test/result( success )
.
