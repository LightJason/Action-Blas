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

import cern.colt.matrix.AbstractMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.lightjason.agentspeak.action.IAction;
import org.lightjason.agentspeak.action.blas.vector.CCreate;
import org.lightjason.agentspeak.error.context.CExecutionIllegealArgumentException;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ITerm;
import org.lightjason.agentspeak.language.execution.IContext;
import org.lightjason.agentspeak.language.execution.IExecution;
import org.lightjason.agentspeak.language.execution.lambda.ILambdaStreaming;
import org.lightjason.agentspeak.testing.IBaseTest;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


/**
 * test math blas vector functions
 */
public final class TestCActionMathBlasVector extends IBaseTest
{

    /**
     * testing vector
     * @note static because of usage in data-provider
     */
    private static final DoubleMatrix1D VECTOR1 = new DenseDoubleMatrix1D( new double[]{2, 5, 3, 8} );

    /**
     * testing vector
     * @note static because of usage in data-provider
     */
    private static final DoubleMatrix1D VECTOR2 = new DenseDoubleMatrix1D( new double[]{8, 6, 2, 1} );


    /**
     * data provider generator
     * @return data
     */
    public static Stream<Arguments> generator()
    {
        final DoubleMatrix1D l_vector1 = new DenseDoubleMatrix1D( new double[]{2, 5, 3, 8} );
        final DoubleMatrix1D l_vector2 = new DenseDoubleMatrix1D( new double[]{8, 6, 2, 1} );

        return Stream.of(

                    Arguments.of( Stream.of( l_vector1, l_vector2 ), CNonZero.class, Stream.of( 4D, 4D ) ),
                    Arguments.of( Stream.of( l_vector1, l_vector2 ), CSum.class, Stream.of( l_vector1.zSum(), l_vector2.zSum() ) ),
                    Arguments.of( Stream.of( l_vector1, l_vector2 ), CSum.class, Stream.of( 60.0 ) )

        );
    }


    /**
     * test all input actions
     *
     * @param p_input input data,
     * @param p_action action
     * @param p_result results
     * @throws IllegalAccessException is thrwon on instantiation error
     * @throws InstantiationException is thrwon on instantiation error
     * @throws NoSuchMethodException is thrwon on instantiation error
     * @throws InvocationTargetException is thrwon on instantiation error
     */
    @ParameterizedTest
    @MethodSource( "generator" )
    public void action( final Stream<ITerm> p_input, final Class<? extends IAction> p_action, final Stream<Object> p_result )
        throws IllegalAccessException, InstantiationException, NoSuchMethodException, InvocationTargetException
    {
        final List<ITerm> l_return = new ArrayList<>();

        p_action.getConstructor().newInstance().execute(
            false, IContext.EMPTYPLAN,
            p_input.map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertArrayEquals(
                l_return.stream().map( ITerm::raw ).toArray(),
                p_result.toArray(),
                p_action.toGenericString()
        );
    }

    /**
     * test create
     */
    @Test
    public void create()
    {
        final List<ITerm> l_return = new ArrayList<>();
        final IExecution l_create = new CCreate();


        l_create.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 2, "dense" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_create.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 4, "sparse" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_create.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 3 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );


        Assertions.assertEquals( 3, l_return.size() );
        Assertions.assertEquals( 2, l_return.get( 0 ).<DoubleMatrix1D>raw().size() );
        Assertions.assertEquals( 4, l_return.get( 1 ).<DoubleMatrix1D>raw().size() );
        Assertions.assertEquals( 3, l_return.get( 2 ).<DoubleMatrix1D>raw().size() );
    }

    /**
     * test set
     */
    @Test
    public void set()
    {
        final DoubleMatrix1D l_vector = new DenseDoubleMatrix1D( 4 );

        new CSet().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 0, 6.0, l_vector ).map( CRawTerm::of ).collect( Collectors.toList() ),
            Collections.emptyList()
        );

        Assertions.assertEquals( 6, l_vector.get( 0 ), 0 );
    }

    /**
     * test toList
     */
    @Test
    public void tolist()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CToList().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( VECTOR1 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( l_return.size(), 1 );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof List );

        final List<Number> l_tolist = l_return.get( 0 ).raw();
        Assertions.assertArrayEquals( Stream.of( 2.0, 5.0, 3.0, 8.0 ).collect( Collectors.toList() ).toArray(), l_tolist.toArray() );
    }

    /**
     * test assign scalar
     */
    @Test
    public void assignscalar()
    {
        final DoubleMatrix1D l_vector = new DenseDoubleMatrix1D( 4 );

        new CAssign().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 2, l_vector ).map( CRawTerm::of ).collect( Collectors.toList() ),
            Collections.emptyList()
        );

        Assertions.assertArrayEquals( Stream.of( 2, 2, 2, 2 ).mapToDouble( i -> i ).toArray(), l_vector.toArray(), 0 );
    }

    /**
     * test assign vector
     */
    @Test
    public void assignvector()
    {
        final DoubleMatrix1D l_vector = new DenseDoubleMatrix1D( 4 );

        new CAssign().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( VECTOR2, l_vector ).map( CRawTerm::of ).collect( Collectors.toList() ),
            Collections.emptyList()
        );

        Assertions.assertArrayEquals( VECTOR2.toArray(), l_vector.toArray(), 0 );
    }

    /**
     * test assign error
     */
    @Test
    public void assignerror()
    {
        Assertions.assertThrows( CExecutionIllegealArgumentException.class,
                                 () -> new CAssign().execute(
                                     false, IContext.EMPTYPLAN,
                                     Stream.of( "xxx", VECTOR1 ).map( CRawTerm::of ).collect( Collectors.toList() ),
                                     Collections.emptyList()
                                )
        );
    }


    /**
     * test get
     */
    @Test
    public void get()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CGet().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( VECTOR1, 0 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof Double );
        Assertions.assertEquals( 2, l_return.get( 0 ).<Double>raw(), 0 );
    }

    /**
     * test copy
     */
    @Test
    public void copy()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CCopy().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( VECTOR1, VECTOR2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 2, l_return.size() );
        Assertions.assertArrayEquals( Stream.of( VECTOR1, VECTOR2 ).toArray(), l_return.stream().map( ITerm::raw ).toArray() );
    }

    /**
     * test parse
     */
    @Test
    public void parse()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CParse().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( "1,2,3", "dense" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        new CParse().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( "4,3,4", "sparse" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( l_return.size(), 2 );
        Assertions.assertArrayEquals( new double[]{1, 2, 3}, l_return.get( 0 ).<DoubleMatrix1D>raw().toArray(), 0 );
        Assertions.assertArrayEquals( new double[]{4, 3, 4}, l_return.get( 1 ).<DoubleMatrix1D>raw().toArray(), 0 );
    }

    /**
     * test fromlist
     */
    @Test
    public void fromlist()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CFromList().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( Stream.of( 1, 2, 3 ).collect( Collectors.toList() ), "dense" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        new CFromList().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( Stream.of( 4, 3, 4 ).collect( Collectors.toList() ), "sparse" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( l_return.size(), 2 );
        Assertions.assertArrayEquals( new double[]{1, 2, 3}, l_return.get( 0 ).<DoubleMatrix1D>raw().toArray(), 0 );
        Assertions.assertArrayEquals( new double[]{4, 3, 4}, l_return.get( 1 ).<DoubleMatrix1D>raw().toArray(), 0 );
    }

    /**
     * test lambda streaming assignable
     */
    @Test
    public void lambdaassignable()
    {
        final ILambdaStreaming<?> l_lambda = new CLambdaStreaming();

        Assertions.assertTrue(
            Stream.of(
                AbstractMatrix1D.class,
                DoubleMatrix1D.class
            ).allMatch( i -> l_lambda.assignable().collect( Collectors.toSet() ).contains( i ) )
        );
    }

    /**
     * test lambda streaming
     */
    @Test
    public void lambda()
    {
        Assertions.assertArrayEquals(
            Stream.of( 2.0, 5.0, 3.0, 8.0 ).toArray(),
            new CLambdaStreaming().apply( VECTOR1 ).toArray()
        );
    }
}
