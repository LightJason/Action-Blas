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

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import org.lightjason.agentspeak.action.IBaseAction;
import org.lightjason.agentspeak.action.blas.EType;
import org.lightjason.agentspeak.common.IPath;
import org.lightjason.agentspeak.error.context.CExecutionException;
import org.lightjason.agentspeak.error.context.CExecutionIllegealArgumentException;
import org.lightjason.agentspeak.language.CCommon;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ITerm;
import org.lightjason.agentspeak.language.execution.IContext;
import org.lightjason.agentspeak.language.fuzzy.IFuzzyValue;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


/**
 * creates a dense- or sparse-matrix of a string.
 * The action parses each argument and returns the matrix object,
 * the last argument can be a string with "dense" or "sparse" to
 * defining a dense or sparse matrix, all other arguments string with
 * a semicolon and space / comma seperated list, the action never fails.
 * Semicolon splits the rows, spaces / comma splits the columns
 *
 * {@code [A|B|C] = .math/blas/matrix/parse("1,2;3,4", "5 6 7; 8 9 10", "dense|sparse" );}
 */
public final class CParse extends IBaseAction
{
    /**
     * serial id
     */
    private static final long serialVersionUID = 1592787131411189781L;
    /**
     * action name
     */
    private static final IPath NAME = namebyclass( CParse.class, "math", "blas", "matrix" );

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
        final List<ITerm> l_arguments = CCommon.flatten( p_argument ).collect( Collectors.toList() );
        final int l_limit;
        final EType l_type;
        if ( CCommon.isssignableto( l_arguments.get( l_arguments.size() - 1 ), String.class )
             && EType.exists( l_arguments.get( l_arguments.size() - 1 ).raw() ) )
        {
            l_type = EType.of( l_arguments.get( l_arguments.size() - 1 ).raw() );
            l_limit = l_arguments.size() - 1;
        }
        else
        {
            l_type = EType.DENSE;
            l_limit = l_arguments.size();
        }


        // create vectors
        switch ( l_type )
        {
            case DENSE:
                l_arguments.stream()
                           .limit( l_limit )
                           .map( ITerm::<String>raw )
                           .map( i -> CParse.parse( i, p_context ) )
                           .map( DenseDoubleMatrix2D::new )
                           .map( CRawTerm::of )
                           .forEach( p_return::add );

                return Stream.of();

            case SPARSE:
                l_arguments.stream()
                           .limit( l_limit )
                           .map( ITerm::<String>raw )
                           .map( i -> CParse.parse( i, p_context ) )
                           .map( SparseDoubleMatrix2D::new )
                           .map( CRawTerm::of )
                           .forEach( p_return::add );

                return Stream.of();

            default:
                throw new CExecutionIllegealArgumentException( p_context, org.lightjason.agentspeak.common.CCommon.languagestring( this, "unknownargument", l_type ) );
        }
    }

    /**
     * parse the string in a list of lists with doubles
     *
     * @param p_string string
     * @param p_context execution context
     * @return 2D double array
     */
    @Nonnull
    private static double[][] parse( @Nonnull final String p_string, @Nonnull final IContext p_context )
    {
        final String[] l_rows = p_string.split( ";" );
        final List<List<Double>> l_matrix = new ArrayList<>();

        final double[][] l_return = new double[l_rows.length][
            Arrays.stream( l_rows )
                  .map( i -> Arrays.stream( i.trim().split( ",|\\s" ) )
                                   .map( String::trim )
                                   .filter( j -> !j.isEmpty() )
                                   .mapToDouble( Double::parseDouble )
                                   .boxed()
                                   .collect( Collectors.toList() )
                  )
                  .mapToInt( i ->
                  {
                      l_matrix.add( i );
                      return i.size();
                  } )
                  .max()
                  .orElseThrow( () -> new CExecutionException( p_context ) )
            ];

        IntStream.range( 0, l_return.length )
                 .boxed()
                 .forEach( i -> IntStream.range( 0, l_return[i].length )
                                         .boxed()
                                         .forEach( j -> l_return[i][j] = l_matrix.get( i ).get( j ) ) );

        return l_return;
    }

}
