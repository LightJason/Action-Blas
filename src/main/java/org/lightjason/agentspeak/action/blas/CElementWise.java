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

import cern.colt.function.tdouble.DoubleDoubleFunction;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.jet.math.tdouble.DoubleFunctions;
import com.codepoetics.protonpack.StreamUtils;
import org.lightjason.agentspeak.action.IBaseAction;
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
 * elementweise vector / matrix operation.
 * The action calculates elementwise different
 * operations (plus, plus-absolute, minus, multiply, divide),
 * all arguments are triples of matrix-operator-matrix|scalar,
 * the action fails on assigning problems
 * {@code [M1|M2|M3] = .math/blas/elementwise( Matrix1, "+", 5, Matrix2, "|+|", Matrix3, Matrix4, "-", 3, [Matrix5, "*", 0.5], [Matrix6, "/", 100]);}
 */
public final class CElementWise extends IBaseAction
{
    /**
     * serial id
     */
    private static final long serialVersionUID = -2655464156364927632L;
    /**
     * action name
     */
    private static final IPath NAME = namebyclass( CElementWise.class, "math", "blas" );

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
        return 3;
    }

    @Nonnull
    @Override
    public Stream<IFuzzyValue<?>> execute( final boolean p_parallel, @Nonnull final IContext p_context,
                                           @Nonnull final List<ITerm> p_argument, @Nonnull final List<ITerm> p_return
    )
    {
        if ( !StreamUtils.windowed(
            CCommon.flatten( p_argument ),
            3,
            3
        ).allMatch( i ->
        {

            switch ( i.get( 1 ).<String>raw().trim() )
            {
                case "+":
                    return CElementWise.apply( i.get( 0 ), i.get( 2 ), DoubleFunctions.plus, ( n, m ) -> n + m, p_return );

                case "|+|":
                    return CElementWise.apply( i.get( 0 ), i.get( 2 ), DoubleFunctions.plusAbs, ( n, m ) -> Math.abs( n + m ), p_return );

                case "-":
                    return CElementWise.apply( i.get( 0 ), i.get( 2 ), DoubleFunctions.minus, ( n, m ) -> n - m, p_return );

                case "*":
                    return CElementWise.apply( i.get( 0 ), i.get( 2 ), DoubleFunctions.mult, ( n, m ) -> n * m, p_return );

                case "/":
                    return CElementWise.apply( i.get( 0 ), i.get( 2 ), DoubleFunctions.div, ( n, m ) -> n / m, p_return );

                default:
                    return false;
            }

        } ) )
            throw new CExecutionIllegalStateException( p_context, org.lightjason.agentspeak.common.CCommon.languagestring( this, "operatorerror" ) );

        return Stream.of();
    }


    /**
     * elementwise assign
     *
     * @param p_left left term argument (matrix argument)
     * @param p_right matrix or scalar value argument
     * @param p_matrixfunction function for matrix-matrix operation
     * @param p_scalarfunction scalar function for value
     * @param p_return return list
     * @return successful executed
     *
     * @note DoubleMatrix1D and DoubleMatrix2D does not use an equal
     * super class for defining the assign method, so code must be
     * created twice for each type
     */
    private static boolean apply( final ITerm p_left, final ITerm p_right,
                                  final DoubleDoubleFunction p_matrixfunction, final BiFunction<Double, Double, Double> p_scalarfunction,
                                  final List<ITerm> p_return
    )
    {
        // operation for matrix
        if ( CCommon.isssignableto( p_left, DoubleMatrix2D.class ) )
        {
            final DoubleMatrix2D l_assign = p_left.<DoubleMatrix2D>raw().copy();

            if ( CCommon.isssignableto( p_right, DoubleMatrix2D.class ) )
            {
                l_assign.assign( p_right.raw(), p_matrixfunction );
                p_return.add( CRawTerm.of( l_assign ) );
                return true;
            }

            if ( CCommon.isssignableto( p_right, Number.class ) )
            {
                l_assign.assign( i -> p_scalarfunction.apply( i, p_right.<Number>raw().doubleValue() ) );
                p_return.add( CRawTerm.of( l_assign ) );
                return true;
            }
        }

        // operation for vector
        if ( CCommon.isssignableto( p_left, DoubleMatrix1D.class ) )
        {
            final DoubleMatrix1D l_assign = p_left.<DoubleMatrix1D>raw().copy();

            if ( CCommon.isssignableto( p_right, DoubleMatrix1D.class ) )
            {
                l_assign.assign( p_right.raw(), p_matrixfunction );
                p_return.add( CRawTerm.of( l_assign ) );
                return true;
            }

            if ( CCommon.isssignableto( p_right, Number.class ) )
            {
                l_assign.assign( i -> p_scalarfunction.apply( i, p_right.<Number>raw().doubleValue() ) );
                p_return.add( CRawTerm.of( l_assign ) );
                return true;
            }
        }

        return false;
    }

}
