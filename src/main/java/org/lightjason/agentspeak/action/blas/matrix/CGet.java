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
import com.codepoetics.protonpack.StreamUtils;
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
import java.util.stream.Collectors;
import java.util.stream.Stream;


/**
 * returns a single element of a matrix.
 * The action returns single elements of a matrix,
 * the first argument is a matrix object, the other
 * arguments are index tuples
 *
 * {@code [A|B] = .math/blas/matrix( Matrix, 3,4, [Row,Column] );}
 */
public final class CGet extends IBaseAction
{
    /**
     * serial id
     */
    private static final long serialVersionUID = -1991625055470044748L;
    /**
     * action name
     */
    private static final IPath NAME = namebyclass( CGet.class, "math", "blas", "matrix" );

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
        final List<ITerm> l_arguments = CCommon.flatten( p_argument ).collect( Collectors.toList() );

        StreamUtils.windowed(
            l_arguments.stream()
                       .skip( 1 )
                       .map( ITerm::<Number>raw )
                       .mapToInt( Number::intValue )
                       .boxed(),
            2
        )
                   .map( i -> l_arguments.get( 0 ).<DoubleMatrix2D>raw().getQuick( i.get( 0 ), i.get( 1 ) ) )
                   .map( CRawTerm::of )
                   .forEach( p_return::add );

        return Stream.of();
    }
}
