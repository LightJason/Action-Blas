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

package org.lightjason.agentspeak.action.blas.matrix;

import cern.colt.matrix.tdouble.DoubleFactory2D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import org.lightjason.agentspeak.action.blas.IBaseAlgebra;
import org.lightjason.agentspeak.common.IPath;
import org.lightjason.agentspeak.language.CCommon;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ITerm;
import org.lightjason.agentspeak.language.execution.IContext;
import org.lightjason.agentspeak.language.fuzzy.IFuzzyValue;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;


/**
 * creates the graph laplacian.
 * For each input adjacency  matrix, the graph laplacian
 * is calculated and returned
 *
 * {@code [L1|L2] = .math/blas/matrix/graphlaplacian( AdjacencyMatrix1, AdjacencyMatrix2 );}
 *
 * @see <a href="https://en.wikipedia.org/wiki/Laplacian_matrix"></a>
 */
public final class CGraphLaplacian extends IBaseAlgebra
{
    /**
     * serial id
     */
    private static final long serialVersionUID = 4781492413860210436L;
    /**
     * action name
     */
    private static final IPath NAME = namebyclass( CGraphLaplacian.class, "math", "blas", "matrix" );

    @Nonnull
    @Override
    public IPath name()
    {
        return NAME;
    }

    @Nonnegative
    @Override
    public int minimalArgumentNumber()
    {
        return 1;
    }

    @Nonnull
    @Override
    public Stream<IFuzzyValue<?>> execute( final boolean p_parallel, @Nonnull final IContext p_context,
                                           @Nonnull final List<ITerm> p_argument, @Nonnull final List<ITerm> p_return
    )
    {
        CCommon.flatten( p_argument )
               .map( ITerm::<DoubleMatrix2D>raw )
               .map( i -> DoubleFactory2D
                   .sparse
                   .diagonal( new DenseDoubleMatrix1D( IntStream.range( 0, i.rows() ).mapToDouble( j -> i.viewRow( j ).cardinality() ).toArray() ) )
                   .assign( i, ( n, m ) -> n - m )
               )
               .map( CRawTerm::of )
               .forEach( p_return::add );

        return Stream.empty();
    }

}
