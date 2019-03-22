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
            ).parallel().allMatch( i ->
            {

                if ( CCommon.isssignableto( i.get( 0 ), DoubleMatrix1D.class ) && CCommon.isssignableto(
                    i.get( 1 ), DoubleMatrix1D.class ) )
                    return CMultiply.<DoubleMatrix1D, DoubleMatrix1D>apply(
                        i.get( 0 ), i.get( 1 ), ( u, v ) -> DENSEALGEBRA.multOuter( u, v, null ), p_return );

                if ( CCommon.isssignableto( i.get( 0 ), DoubleMatrix2D.class ) && CCommon.isssignableto(
                    i.get( 1 ), DoubleMatrix2D.class ) )
                    return CMultiply.<DoubleMatrix2D, DoubleMatrix2D>apply( i.get( 0 ), i.get( 1 ), DENSEALGEBRA::mult, p_return );

                if ( CCommon.isssignableto( i.get( 0 ), DoubleMatrix2D.class ) && CCommon.isssignableto(
                    i.get( 1 ), DoubleMatrix1D.class ) )
                    return CMultiply.<DoubleMatrix2D, DoubleMatrix1D>apply( i.get( 0 ), i.get( 1 ), DENSEALGEBRA::mult, p_return );

                return CCommon.isssignableto( i.get( 0 ), DoubleMatrix1D.class ) && CCommon.isssignableto(
                    i.get( 1 ), DoubleMatrix2D.class )
                       && CMultiply.<DoubleMatrix1D, DoubleMatrix2D>apply(
                    i.get( 0 ), i.get( 1 ), ( u, v ) -> DENSEALGEBRA.mult( v, u ), p_return );

            } )
        )
            throw new CExecutionIllegalStateException( p_context, org.lightjason.agentspeak.common.CCommon.languagestring( this, "operatorerror" ) );

        return Stream.of();
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
