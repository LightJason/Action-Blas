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

package org.lightjason.agentspeak.action.blas;

import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import com.codepoetics.protonpack.StreamUtils;
import org.lightjason.agentspeak.common.IPath;
import org.lightjason.agentspeak.error.context.CExecutionIllegalStateException;
import org.lightjason.agentspeak.language.CCommon;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ITerm;
import org.lightjason.agentspeak.language.execution.IContext;
import org.lightjason.agentspeak.language.fuzzy.IFuzzyValue;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Stream;


/**
 * defines matrix- / vector-products.
 * The action multiplies tupel-wise all unflatten arguments,
 * the action fails iif the multiply cannot executed e.g. on wrong
 * input
 *
 * {@code [M1|M2|M3] = .math/blas/multiply( Vector1, Vector2, [[Matrix1, Matrix2], Matrix3, Vector3] );}
 */
public final class CMultiply extends IBaseAlgebra
{
    /**
     * serial id
     */
    private static final long serialVersionUID = 7399930315943440254L;
    /**
     * action name
     */
    private static final IPath NAME = namebyclass( CMultiply.class, "math", "blas" );

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
        return 2;
    }

    @Nonnull
    @Override
    public Stream<IFuzzyValue<?>> execute( final boolean p_parallel, @Nonnull final IContext p_context,
                                           @Nonnull final List<ITerm> p_argument, @Nonnull final List<ITerm> p_return )
    {
        if ( !StreamUtils.windowed(
                CCommon.flatten( p_argument ),
                2,
                2
            ).parallel().allMatch( i -> CCommon.streamconcatstrict(
            cast( DoubleMatrix1D.class, DoubleMatrix1D.class, i.get( 0 ), i.get( 1 ), ( u, v ) -> DENSEALGEBRA.multOuter( u, v, null ), p_return ),
            cast( DoubleMatrix2D.class, DoubleMatrix2D.class, i.get( 0 ), i.get( 1 ), DENSEALGEBRA::mult, p_return ),
            cast( DoubleMatrix2D.class, DoubleMatrix1D.class, i.get( 0 ), i.get( 1 ), DENSEALGEBRA::mult, p_return ),
            cast( DoubleMatrix1D.class, DoubleMatrix2D.class, i.get( 0 ), i.get( 1 ), ( u, v ) -> DENSEALGEBRA.mult( v, u ), p_return )
            ).findFirst().orElse( false )
        ) )
            throw new CExecutionIllegalStateException( p_context, org.lightjason.agentspeak.common.CCommon.languagestring( this, "operatorerror" ) );

        return Stream.of();
    }

    /**
     * execute with casting
     *
     * @param p_lhsclass left-hand-side class
     * @param p_rhsclass right-hand-side class
     * @param p_lhs left-hand-side
     * @param p_rhs right-hand-side
     * @param p_function executoin function
     * @param p_return return value
     * @tparam U left-hand-side type
     * @tparam V right-hand-side type
     * @return successfully flag
     */
    private static <U, V> Stream<Boolean> cast( @Nonnull final Class<U> p_lhsclass, @Nonnull final Class<V> p_rhsclass,
                                                @Nonnull final ITerm p_lhs, @Nonnull final ITerm p_rhs,
                                                @Nonnull final BiFunction<U, V, ?> p_function, @Nonnull final List<ITerm> p_return )
    {
        return CCommon.isssignableto( p_lhs, p_lhsclass ) && CCommon.isssignableto( p_rhs, p_rhsclass )
               ? Stream.of( CMultiply.<U, V>apply( p_lhs, p_rhs, p_function, p_return ) )
               : Stream.of();
    }

    /**
     * apply method
     *
     * @param p_left first element
     * @param p_right second element
     * @param p_function function for the two elements
     * @param p_return return list
     * @return successful executed
     *
     * @tparam U first argument type
     * @tparam V second argument type
     */
    private static <U, V> boolean apply( final ITerm p_left, final ITerm p_right, final BiFunction<U, V, ?> p_function, final List<ITerm> p_return )
    {
        p_return.add( CRawTerm.of( p_function.apply( p_left.raw(), p_right.raw() ) ) );
        return true;
    }

}
