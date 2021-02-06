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

import cern.colt.matrix.AbstractMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix1D;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.SparseDoubleMatrix2D;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.lightjason.agentspeak.action.IAction;
import org.lightjason.agentspeak.action.blas.matrix.CAssign;
import org.lightjason.agentspeak.action.blas.matrix.CColumn;
import org.lightjason.agentspeak.action.blas.matrix.CColumnSum;
import org.lightjason.agentspeak.action.blas.matrix.CColumns;
import org.lightjason.agentspeak.action.blas.matrix.CCondition;
import org.lightjason.agentspeak.action.blas.matrix.CCopy;
import org.lightjason.agentspeak.action.blas.matrix.CCreate;
import org.lightjason.agentspeak.action.blas.matrix.CDeterminant;
import org.lightjason.agentspeak.action.blas.matrix.CDiagonal;
import org.lightjason.agentspeak.action.blas.matrix.CDimension;
import org.lightjason.agentspeak.action.blas.matrix.CEigen;
import org.lightjason.agentspeak.action.blas.matrix.CGet;
import org.lightjason.agentspeak.action.blas.matrix.CGraphLaplacian;
import org.lightjason.agentspeak.action.blas.matrix.CIdentity;
import org.lightjason.agentspeak.action.blas.matrix.CInfinityNorm;
import org.lightjason.agentspeak.action.blas.matrix.CInvert;
import org.lightjason.agentspeak.action.blas.matrix.CLambdaStreaming;
import org.lightjason.agentspeak.action.blas.matrix.CMatrixNorm;
import org.lightjason.agentspeak.action.blas.matrix.CNonZero;
import org.lightjason.agentspeak.action.blas.matrix.CNormalizedGraphLaplacian;
import org.lightjason.agentspeak.action.blas.matrix.COneNorm;
import org.lightjason.agentspeak.action.blas.matrix.CParse;
import org.lightjason.agentspeak.action.blas.matrix.CPerronFrobenius;
import org.lightjason.agentspeak.action.blas.matrix.CPower;
import org.lightjason.agentspeak.action.blas.matrix.CRank;
import org.lightjason.agentspeak.action.blas.matrix.CRow;
import org.lightjason.agentspeak.action.blas.matrix.CRowSum;
import org.lightjason.agentspeak.action.blas.matrix.CRows;
import org.lightjason.agentspeak.action.blas.matrix.CSet;
import org.lightjason.agentspeak.action.blas.matrix.CSingularValue;
import org.lightjason.agentspeak.action.blas.matrix.CSolve;
import org.lightjason.agentspeak.action.blas.matrix.CSubMatrix;
import org.lightjason.agentspeak.action.blas.matrix.CSum;
import org.lightjason.agentspeak.action.blas.matrix.CToList;
import org.lightjason.agentspeak.action.blas.matrix.CTrace;
import org.lightjason.agentspeak.action.blas.matrix.CTranspose;
import org.lightjason.agentspeak.action.blas.matrix.CTwoNorm;
import org.lightjason.agentspeak.error.context.CExecutionIllegealArgumentException;
import org.lightjason.agentspeak.language.CRawTerm;
import org.lightjason.agentspeak.language.ITerm;
import org.lightjason.agentspeak.language.execution.IContext;
import org.lightjason.agentspeak.language.execution.IExecution;
import org.lightjason.agentspeak.language.execution.lambda.ILambdaStreaming;
import org.lightjason.agentspeak.testing.IBaseTest;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * test math blas matrix functions
 */
public final class TestCActionMathBlasMatrix extends IBaseTest
{

    /**
     * testing matrix
     * @note static because of usage in data-provider
     */
    private static final DoubleMatrix2D MATRIX1 = new DenseDoubleMatrix2D( new double[][]{{2, 6}, {3, 8}} );

    /**
     * testing matrix
     * @note static because of usage in data-provider
     */
    private static final DoubleMatrix2D MATRIX2 = new DenseDoubleMatrix2D( new double[][]{{2, 2}, {3, 1}} );

    /**
     * data provider generator
     * @return data
     */
    public static Stream<Arguments> generator()
    {
        final DoubleMatrix2D l_matrix1 = new DenseDoubleMatrix2D( new double[][]{{2, 6}, {3, 8}} );
        final DoubleMatrix2D l_matrix2 = new DenseDoubleMatrix2D( new double[][]{{2, 2}, {3, 1}} );


        return Stream.of(

                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CColumns.class, Stream.of( 2D, 2D ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CDimension.class, Stream.of( 2D, 2D, 2D, 2D ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CRows.class, Stream.of( 2D, 2D ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CNonZero.class, Stream.of( 4D, 4D ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CNonZero.class, Stream.of( 4D, 4D ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CCondition.class, Stream.of( 56.48229533707812, 4.265564437074639 ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CDeterminant.class, Stream.of( -2.000000000000001, -4.0 ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CTwoNorm.class, Stream.of( 10.628480167651258, 4.130648586880582 ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), COneNorm.class, Stream.of( 14.0000, 5.0000 ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CMatrixNorm.class, Stream.of( 10.63014581273465, 4.242640687119285 ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CInfinityNorm.class, Stream.of( 11.0000, 4.0000 ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CRank.class, Stream.of( 2D, 2D ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CSum.class, Stream.of( l_matrix1.zSum(), l_matrix2.zSum() ) ),
                Arguments.of( Stream.of( l_matrix1, l_matrix2 ), CTrace.class, Stream.of( 10.0, 3.0 ) )
        );
    }


    /**
     * test all input actions
     *
     * @param p_input tripel of input data, actions and results
     * @throws IllegalAccessException is thrwon on instantiation error
     * @throws InstantiationException is thrwon on instantiation error
     * @throws NoSuchMethodException is thrwon on instantiation error
     * @throws InvocationTargetException is thrwon on instantiation error
     */
    @ParameterizedTest
    @MethodSource( "generator" )
    public void action( final Stream<Object> p_input, final Class<? extends IAction> p_action, final Stream<Object> p_result )
        throws IllegalAccessException, InstantiationException, NoSuchMethodException, InvocationTargetException
    {
        final List<ITerm> l_return = new ArrayList<>();

        p_action.getConstructor().newInstance().execute(
            false, IContext.EMPTYPLAN,
            p_input.map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertArrayEquals(
                p_result.toArray(),
                l_return.stream().map( ITerm::raw ).toArray(),
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
            false,
            IContext.EMPTYPLAN,
            Stream.of( 2, 2, "dense" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_create.execute(
            false,
            IContext.EMPTYPLAN,
            Stream.of( 2, 3, "sparse" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_create.execute(
            false,
            IContext.EMPTYPLAN,
            Stream.of( 4, 4 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );


        Assertions.assertEquals( 3, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertTrue( l_return.get( 1 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertTrue( l_return.get( 2 ).raw() instanceof DoubleMatrix2D );

        Assertions.assertEquals( 4, l_return.get( 0 ).<DoubleMatrix2D>raw().size() );
        Assertions.assertEquals( 2, l_return.get( 0 ).<DoubleMatrix2D>raw().rows() );
        Assertions.assertEquals( 2, l_return.get( 0 ).<DoubleMatrix2D>raw().columns() );

        Assertions.assertEquals( 6, l_return.get( 1 ).<DoubleMatrix2D>raw().size() );
        Assertions.assertEquals( 2, l_return.get( 1 ).<DoubleMatrix2D>raw().rows() );
        Assertions.assertEquals( 3, l_return.get( 1 ).<DoubleMatrix2D>raw().columns() );

        Assertions.assertEquals( 16, l_return.get( 2 ).<DoubleMatrix2D>raw().size() );
        Assertions.assertEquals( 4, l_return.get( 2 ).<DoubleMatrix2D>raw().rows() );
        Assertions.assertEquals( 4, l_return.get( 2 ).<DoubleMatrix2D>raw().columns() );
    }

    /**
     * test column
     */
    @Test
    public void column()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CColumn().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 1, MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix1D );
        Assertions.assertEquals( 2, l_return.get( 0 ).<DoubleMatrix1D>raw().size() );

        Assertions.assertEquals( 2, l_return.get( 0 ).<DoubleMatrix1D>raw().get( 0 ), 0 );
        Assertions.assertEquals( 1, l_return.get( 0 ).<DoubleMatrix1D>raw().get( 1 ), 0 );
    }

    /**
     * test row
     */
    @Test
    public void row()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CRow().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 1, MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix1D );
        Assertions.assertEquals( 2, l_return.get( 0 ).<DoubleMatrix1D>raw().size() );

        Assertions.assertEquals( 3, l_return.get( 0 ).<DoubleMatrix1D>raw().get( 0 ), 0 );
        Assertions.assertEquals( 1, l_return.get( 0 ).<DoubleMatrix1D>raw().get( 1 ), 0 );
    }

    /**
     * test power
     */
    @Test
    public void power()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CPower().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 2, MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertEquals( 4, l_return.get( 0 ).<DoubleMatrix2D>raw().size() );

        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{10.0, 6.0}, {9.0, 7.0}} ).toArray(),
            l_return.get( 0 ).<DoubleMatrix2D>raw().toArray()
        );

    }

    /**
     * test set
     */
    @Test
    public void set()
    {
        new CSet().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 0, 1, 6.0, MATRIX1 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            Collections.emptyList()
        );

        Assertions.assertEquals( 6, MATRIX1.get( 0, 1 ), 0 );
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
            Stream.of( MATRIX1 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof List );

        Assertions.assertArrayEquals(
            Stream.of( 2.0, 6.0, 3.0, 8.0 ).collect( Collectors.toList() ).toArray(),
            l_return.get( 0 ).<List<?>>raw().toArray()
        );
    }

    /**
     * test transpose
     */
    @Test
    public void transpose()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CTranspose().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );

        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{2.0, 3.0}, {2.0, 1.0}} ).toArray(),
            l_return.get( 0 ).<DoubleMatrix2D>raw().toArray()
        );
    }

    /**
     * test submatrix
     */
    @Test
    public void submatrix()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CSubMatrix().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 0, 0, 0, 1, MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );

        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{2.0, 2.0}} ).toArray(),
            l_return.get( 0 ).<DoubleMatrix2D>raw().toArray()
        );
    }

    /**
     * test solve
     */
    @Test
    public void solve()
    {
        final List<ITerm> l_return = new ArrayList<>();
        final IExecution l_solve = new CSolve();

        l_solve.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX1, MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_solve.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX1, new DenseDoubleMatrix1D( new double[]{2, 3} ) ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 2, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertTrue( l_return.get( 1 ).raw() instanceof DoubleMatrix2D );

        Assertions.assertArrayEquals(
            new double[][]{{1.0, -4.999999999999998}, {0.0, 1.9999999999999993}},
            l_return.get( 0 ).<DoubleMatrix2D>raw().toArray()
        );

        Assertions.assertArrayEquals(
            new double[][]{{1.0}, {0.0}},
            l_return.get( 1 ).<DoubleMatrix2D>raw().toArray()
        );
    }

    /**
     * test perron-frobenius
     */
    @Test
    public void perronfrobenius()
    {
        final List<ITerm> l_return = new ArrayList<>();
        final DoubleMatrix2D l_matrix = new DenseDoubleMatrix2D( new double[][]{{0.1, 0.5, 0.3}, {0.5, 0.1, 0.3}, {0.3, 0.3, 0.1}} );

        new CPerronFrobenius().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 5, l_matrix ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertArrayEquals( new double[]{0.614167, 0.613706, 0.496149}, l_return.get( 0 ).<DoubleMatrix1D>raw().toArray(), 0.1 );
    }

    /**
     * test assign scalar
     */
    @Test
    public void assignscalar()
    {
        final DoubleMatrix2D l_matrix = new DenseDoubleMatrix2D( 2, 2 );

        new CAssign().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( 2, l_matrix ).map( CRawTerm::of ).collect( Collectors.toList() ),
            Collections.emptyList()
        );

        Assertions.assertArrayEquals( new double[][]{{2.0, 2.0}, {2.0, 2.0}}, l_matrix.toArray() );
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
                                     Stream.of( "xx", MATRIX1 ).map( CRawTerm::of ).collect( Collectors.toList() ),
                                     Collections.emptyList()
                                 )
        );
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
                AbstractMatrix2D.class,
                DoubleMatrix2D.class
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
            Stream.of( 2.0, 3.0, 6.0, 8.0 ).toArray(),
            new CLambdaStreaming().apply( MATRIX1 ).toArray()
        );
    }

    /**
     * test assign matrix
     */
    @Test
    public void assignmatrix()
    {
        final DoubleMatrix2D l_matrix = new DenseDoubleMatrix2D( 2, 2 );

        new CAssign().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX2, l_matrix ).map( CRawTerm::of ).collect( Collectors.toList() ),
            Collections.emptyList()
        );

        Assertions.assertArrayEquals( l_matrix.toArray(), MATRIX2.toArray() );
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
            Stream.of( MATRIX2, 0, 1 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof Double );
        Assertions.assertEquals( 2, l_return.get( 0 ).<Double>raw(), 0 );
    }

    /**
     * test parse
     */
    @Test
    public void parse()
    {
        final List<ITerm> l_return = new ArrayList<>();
        final IExecution l_parse = new CParse();

        l_parse.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( "1,2;3,4", "dense" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_parse.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( "4,3;2,1;0,0", "sparse" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        l_parse.execute(
            false, IContext.EMPTYPLAN,
            Stream.of( "1;1;1" ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );



        Assertions.assertEquals( 3, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertTrue( l_return.get( 1 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertTrue( l_return.get( 2 ).raw() instanceof DoubleMatrix2D );

        Assertions.assertArrayEquals(
            new double[][]{{1.0, 2.0}, {3.0, 4.0}},
            l_return.get( 0 ).<DoubleMatrix2D>raw().toArray()
        );
        Assertions.assertArrayEquals(
            new double[][]{{4.0, 3.0}, {2.0, 1.0}, {0.0, 0.0}},
            l_return.get( 1 ).<DoubleMatrix2D>raw().toArray()
        );
        Assertions.assertArrayEquals(
            new double[][]{{1.0}, {1.0}, {1.0}},
            l_return.get( 2 ).<DoubleMatrix2D>raw().toArray()
        );
    }

    /**
     * test invert
     */
    @Test
    public void invert()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CInvert().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{-0.24999999999999994, 0.5}, {0.7499999999999999, -0.4999999999999999}} ).toArray(),
            l_return.get( 0 ).<DoubleMatrix2D>raw().toArray()
        );
    }

    /**
     * test eigen
     */
    @Test
    public void eigen()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CEigen().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 2, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DenseDoubleMatrix1D );
        Assertions.assertArrayEquals( l_return.get( 0 ).<DenseDoubleMatrix1D>raw().toArray(), Stream.of( 4, -1 ).mapToDouble( i -> i ).toArray(), 0 );

        Assertions.assertTrue( l_return.get( 1 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{0.7071067811865475, -0.565685424949238}, {0.7071067811865475, 0.8485281374238569}} ).toArray(),
            l_return.get( 1 ).<DoubleMatrix2D>raw().toArray()
        );
    }

    /**
     * test singularvalue
     */
    @Test
    public void singularvalue()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CSingularValue().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 3, l_return.size() );
        Assertions.assertTrue( l_return.get( 0 ).raw() instanceof DenseDoubleMatrix1D );
        Assertions.assertArrayEquals(
            l_return.get( 0 ).<DenseDoubleMatrix1D>raw().toArray(),
            Stream.of(  4.130648586880582, 0.9683709267122025 ).mapToDouble( i -> i ).toArray(), 0 );

        Assertions.assertTrue( l_return.get( 1 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{-0.6618025632357403, -0.7496781758158657}, {-0.7496781758158659, 0.66180256323574}} ).toArray(),
            l_return.get( 1 ).<DoubleMatrix2D>raw().toArray()
        );

        Assertions.assertTrue( l_return.get( 2 ).raw() instanceof DoubleMatrix2D );
        Assertions.assertArrayEquals(
            new DenseDoubleMatrix2D( new double[][]{{-0.8649100931185951, 0.5019268181932333}, {-0.5019268181932333, -0.8649100931185951}} ).toArray(),
            l_return.get( 2 ).<DoubleMatrix2D>raw().toArray()
        );
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
            Stream.of( MATRIX1, MATRIX2 ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 2, l_return.size() );
        Assertions.assertArrayEquals( Stream.of( MATRIX1, MATRIX2 ).toArray(), l_return.stream().map( ITerm::raw ).toArray() );
    }


    /**
     * test graph-laplacian
     */
    @Test
    public void graphlaplacian()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CGraphLaplacian().execute(
            false, IContext.EMPTYPLAN,
            Stream.of(
                    new SparseDoubleMatrix2D( new double[][]{
                        {0, 1, 0, 0, 1, 0},
                        {1, 0, 1, 0, 1, 0},
                        {0, 1, 0, 1, 0, 0},
                        {0, 0, 1, 0, 1, 1},
                        {1, 1, 0, 1, 0, 0},
                        {0, 0, 0, 1, 0, 0}
                    } )
                ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        final DoubleMatrix2D l_result = l_return.get( 0 ).raw();

        IntStream.range( 0, l_result.rows() )
                 .boxed()
                 .map( l_result::viewRow )
                 .mapToDouble( DoubleMatrix1D::zSum )
                 .forEach( i -> Assertions.assertEquals( 0, i, 0 ) );

        IntStream.range( 0, l_result.columns() )
                 .boxed()
                 .map( l_result::viewColumn )
                 .mapToDouble( DoubleMatrix1D::zSum )
                 .forEach( i -> Assertions.assertEquals( 0, i, 0 ) );
    }


    /**
     * test normalized graph-laplacian
     */
    @Test
    public void normalizedgraphlaplacian()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CNormalizedGraphLaplacian().execute(
            false, IContext.EMPTYPLAN,
            Stream.of(
                new SparseDoubleMatrix2D( new double[][]{
                    {0, 1, 0, 0, 1, 0},
                    {1, 0, 1, 0, 1, 0},
                    {0, 1, 0, 1, 0, 0},
                    {0, 0, 1, 0, 1, 1},
                    {1, 1, 0, 1, 0, 0},
                    {0, 0, 0, 1, 0, 0}
                } )
            ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        final DoubleMatrix2D l_result = l_return.get( 0 ).raw();

        IntStream.range( 0, l_result.rows() ).boxed().forEach( i -> Assertions.assertEquals( 1, l_result.getQuick( i, i ), 0 ) );
        IntStream.range( 0, l_result.rows() )
                 .boxed()
                 .map( l_result::viewRow )
                 .mapToDouble( DoubleMatrix1D::zSum )
                 .forEach( i -> Assertions.assertEquals( 0, i, 1e-10 ) );
    }

    /**
     * test row sum
     */
    @Test
    public void rowsum()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CRowSum().execute(
            false, IContext.EMPTYPLAN,
            Stream.of(
                new SparseDoubleMatrix2D( new double[][]{
                    {1, 0, 0, 0, 0, 0},
                    {1, 2, 0, 0, 0, 0},
                    {1, 2, 3, 0, 0, 0},
                    {1, 2, 3, 4, 0, 0},
                    {1, 2, 3, 4, 5, 0},
                    {1, 2, 3, -1, -2, -3}
                } )
            ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertArrayEquals(
            Stream.of( 1D, 3D, 6D, 10D, 15D, 0D ).toArray(),
            Arrays.stream( l_return.get( 0 ).<DoubleMatrix1D>raw().toArray() ).boxed().toArray()
        );
    }


    /**
     * test column sum
     */
    @Test
    public void columsum()
    {
        final List<ITerm> l_return = new ArrayList<>();

        new CColumnSum().execute(
            false, IContext.EMPTYPLAN,
            Stream.of(
                new SparseDoubleMatrix2D( new double[][]{
                    {1, 0, 0, 0, 0, 0},
                    {1, 2, 0, 0, 0, 0},
                    {1, 2, 3, 0, 0, 0},
                    {1, 2, 3, 4, 0, 0},
                    {1, 2, 3, 4, 5, 0},
                    {1, 2, 3, -1, -2, -3}
                } )
            ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        Assertions.assertArrayEquals(
            Stream.of( 6D, 10D, 12D, 7D, 3D, -3D ).toArray(),
            Arrays.stream( l_return.get( 0 ).<DoubleMatrix1D>raw().toArray() ).boxed().toArray()
        );
    }


    /**
     * test identity
     */
    @Test
    public void identity()
    {
        final int l_size = Math.abs( new Random().nextInt( 98 ) + 2 );
        final List<ITerm> l_return = new ArrayList<>();

        new CIdentity().execute(
            false, IContext.EMPTYPLAN,
            Stream.of( l_size ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        final DoubleMatrix2D l_result = l_return.get( 0 ).raw();

        Assertions.assertTrue(
            IntStream.range( 0, l_result.rows() )
                 .boxed()
                 .flatMap( i -> IntStream.range( 0, l_result.columns() )
                                         .boxed()
                                         .map( j -> i.equals( j ) ? l_result.getQuick( i, j ) == 1D : l_result.getQuick( i, j ) == 0D )
                 )
            .allMatch( i -> i )
        );
    }


    /**
     * test diagonal
     */
    @Test
    public void diagonal()
    {
        final List<ITerm> l_return = new ArrayList<>();
        final double[] l_data = new double[]{1, 3, 5, 11};

        new CDiagonal().execute(
            false, IContext.EMPTYPLAN,
            Stream.of(
                new DenseDoubleMatrix1D( l_data )
            ).map( CRawTerm::of ).collect( Collectors.toList() ),
            l_return
        );

        Assertions.assertEquals( 1, l_return.size() );
        final DoubleMatrix2D l_result = l_return.get( 0 ).raw();

        Assertions.assertArrayEquals(
            IntStream.range( 0, l_result.rows() )
                     .boxed()
                     .map( i -> l_result.getQuick( i, i ) )
                     .toArray(),
            Arrays.stream( l_data ).boxed().toArray()
        );

        IntStream.range( 0, l_result.rows() )
                 .forEach( i -> IntStream.range( 0, l_result.columns() )
                                         .filter( j -> i != j )
                                         .forEach( j -> Assertions.assertEquals( 0, l_result.getQuick( i, j ), 0 ) ) );
    }

}
