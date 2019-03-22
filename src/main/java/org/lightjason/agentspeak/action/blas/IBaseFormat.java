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

import cern.colt.matrix.tdouble.algo.DoubleFormatter;
import org.lightjason.agentspeak.action.generic.CPrint;


/**
 * base formatter of blas structures
 */
public abstract class IBaseFormat<T> extends CPrint.IFormatter<T>
{
    /**
     * formatter definition
     */
    protected static final DoubleFormatter FORMATTER;
    /**
     * serial id
     */
    private static final long serialVersionUID = -5600570343858338053L;

    static
    {
        FORMATTER = new DoubleFormatter();
        FORMATTER.setRowSeparator( "; " );
        FORMATTER.setColumnSeparator( "," );
        FORMATTER.setPrintShape( false );
    }

}
