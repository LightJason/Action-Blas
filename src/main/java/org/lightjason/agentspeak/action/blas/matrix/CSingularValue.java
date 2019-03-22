/*
 * @cond LICENSE
 * ######################################################################################
 * # LGPL License                                                                       #
 * #                                                                                    #
 * # This file is part of the LightJason AgentSpeak(L++)                                #
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

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.algo.decomposition.DenseDoubleSingularValueDecomposition;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import org.lightjason.agentspeak.action.IBaseAction;
import org.lightjason.agentspeak.common.IPath;
import org.lightjason.agentspeak.language.CCommon;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ITerm;
import org.lightjason.agentspeak.language.execution.IContext;
import org.lightjason.agentspeak.language.fuzzy.IFuzzyValue;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import java.util.List;
import java.util.stream.Stream;


/**
 * creates the singular value decomposition of a matrix.
 * For each input matrix the singular value decompisition is
 * called and the values, and the two matrixes (left / right)
 * are returned
 *
 * {@code [Values1|U1|V1|Values2|U2|V2] = .blas/matrix/singularvalue(Matrix1, Matrix2);}
 *
 * @see https://en.wikipedia.org/wiki/Singular_value_decomposition
 */
public final class CSingularValue extends IBaseAction
{
    /**
     * serial id
     */
    private static final long serialVersionUID = 3316611747769800455L;
    /**
     * action name
     */
    private static final IPath NAME = namebyclass( CSingularValue.class, "math", "blas", "matrix" );

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
               .map( i -> new DenseDoubleSingularValueDecomposition( i, true, false ) )
               .forEach( i ->
               {
                   p_return.add( CRawTerm.of( new DenseDoubleMatrix1D( i.getSingularValues() ) ) );
                   p_return.add( CRawTerm.of( i.getU() ) );
                   p_return.add( CRawTerm.of( i.getV() ) );
               } );

        return Stream.of();
    }
}
