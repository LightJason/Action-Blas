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

import cern.colt.matrix.tdouble.DoubleMatrix2D;

import java.text.MessageFormat;


/**
 * formatter of 2D matrix
 */
public final class CFormat2D extends IBaseFormat<DoubleMatrix2D>
{
    /**
     * serial id
     */
    private static final long serialVersionUID = 2504711213170928363L;

    @Override
    protected Class<?> getType()
    {
        return DoubleMatrix2D.class;
    }

    @Override
    protected String format( final DoubleMatrix2D p_data )
    {
        return MessageFormat.format( "[{0}x{1}]({2})", p_data.rows(), p_data.columns(), FORMATTER.toString( p_data ) );
    }
}
