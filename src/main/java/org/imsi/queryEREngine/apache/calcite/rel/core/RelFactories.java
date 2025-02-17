/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.imsi.queryEREngine.apache.calcite.rel.core;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.SortedSet;

import javax.annotation.Nonnull;

import org.apache.calcite.linq4j.function.Experimental;
import org.imsi.queryEREngine.apache.calcite.plan.Context;
import org.imsi.queryEREngine.apache.calcite.plan.Contexts;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.plan.ViewExpanders;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelDistribution;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.hint.RelHint;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalAggregate;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalCorrelate;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalExchange;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalFilter;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalIntersect;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalJoin;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalMatch;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalMinus;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalRepeatUnion;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalSnapshot;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalSort;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalSortExchange;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableFunctionScan;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableScan;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableSpool;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalUnion;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalValues;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelColumnMapping;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexLiteral;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.schema.TranslatableTable;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableBitSet;
import org.imsi.queryEREngine.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

/**
 * Contains factory interface and default implementation for creating various
 * rel nodes.
 */
public class RelFactories {
	public static final ProjectFactory DEFAULT_PROJECT_FACTORY =
			new ProjectFactoryImpl();

	public static final FilterFactory DEFAULT_FILTER_FACTORY =
			new FilterFactoryImpl();

	public static final JoinFactory DEFAULT_JOIN_FACTORY = new JoinFactoryImpl();

	public static final CorrelateFactory DEFAULT_CORRELATE_FACTORY =
			new CorrelateFactoryImpl();

	public static final SortFactory DEFAULT_SORT_FACTORY =
			new SortFactoryImpl();

	public static final ExchangeFactory DEFAULT_EXCHANGE_FACTORY =
			new ExchangeFactoryImpl();

	public static final SortExchangeFactory DEFAULT_SORT_EXCHANGE_FACTORY =
			new SortExchangeFactoryImpl();

	public static final AggregateFactory DEFAULT_AGGREGATE_FACTORY =
			new AggregateFactoryImpl();

	public static final MatchFactory DEFAULT_MATCH_FACTORY =
			new MatchFactoryImpl();

	public static final SetOpFactory DEFAULT_SET_OP_FACTORY =
			new SetOpFactoryImpl();

	public static final ValuesFactory DEFAULT_VALUES_FACTORY =
			new ValuesFactoryImpl();

	public static final TableScanFactory DEFAULT_TABLE_SCAN_FACTORY =
			new TableScanFactoryImpl();

	public static final TableFunctionScanFactory
	DEFAULT_TABLE_FUNCTION_SCAN_FACTORY = new TableFunctionScanFactoryImpl();

	public static final SnapshotFactory DEFAULT_SNAPSHOT_FACTORY =
			new SnapshotFactoryImpl();

	public static final SpoolFactory DEFAULT_SPOOL_FACTORY =
			new SpoolFactoryImpl();

	public static final RepeatUnionFactory DEFAULT_REPEAT_UNION_FACTORY =
			new RepeatUnionFactoryImpl();

	public static final Struct DEFAULT_STRUCT =
			new Struct(DEFAULT_FILTER_FACTORY,
					DEFAULT_PROJECT_FACTORY,
					DEFAULT_AGGREGATE_FACTORY,
					DEFAULT_SORT_FACTORY,
					DEFAULT_EXCHANGE_FACTORY,
					DEFAULT_SORT_EXCHANGE_FACTORY,
					DEFAULT_SET_OP_FACTORY,
					DEFAULT_JOIN_FACTORY,
					DEFAULT_CORRELATE_FACTORY,
					DEFAULT_VALUES_FACTORY,
					DEFAULT_TABLE_SCAN_FACTORY,
					DEFAULT_TABLE_FUNCTION_SCAN_FACTORY,
					DEFAULT_SNAPSHOT_FACTORY,
					DEFAULT_MATCH_FACTORY,
					DEFAULT_SPOOL_FACTORY,
					DEFAULT_REPEAT_UNION_FACTORY);

	/** A {@link RelBuilderFactory} that creates a {@link RelBuilder} that will
	 * create logical relational expressions for everything. */
	public static final RelBuilderFactory LOGICAL_BUILDER =
			RelBuilder.proto(Contexts.of(DEFAULT_STRUCT));

	private RelFactories() {
	}

	/**
	 * Can create a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject} of the
	 * appropriate type for this rule's calling convention.
	 */
	public interface ProjectFactory {
		/**
		 * Creates a project.
		 *
		 * @param input The input
		 * @param hints The hints
		 * @param childExprs The projection expressions
		 * @param fieldNames The projection field names
		 * @return a project
		 */
		RelNode createProject(RelNode input, List<RelHint> hints,
				List<? extends RexNode> childExprs, List<String> fieldNames);

		@Deprecated // to be removed before 1.23
		default RelNode createProject(RelNode input,
				List<? extends RexNode> childExprs, List<String> fieldNames) {
			return createProject(input, ImmutableList.of(), childExprs, fieldNames);
		}
	}

	/**
	 * Implementation of {@link ProjectFactory} that returns a vanilla
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject}.
	 */
	private static class ProjectFactoryImpl implements ProjectFactory {
		@Override
		public RelNode createProject(RelNode input, List<RelHint> hints,
				List<? extends RexNode> childExprs, List<String> fieldNames) {
			return LogicalProject.create(input, hints, childExprs, fieldNames);
		}
	}

	/**
	 * Can create a {@link Sort} of the appropriate type
	 * for this rule's calling convention.
	 */
	public interface SortFactory {
		/** Creates a sort. */
		RelNode createSort(RelNode input, RelCollation collation, RexNode offset,
				RexNode fetch);

		@Deprecated // to be removed before 2.0
		default RelNode createSort(RelTraitSet traitSet, RelNode input,
				RelCollation collation, RexNode offset, RexNode fetch) {
			return createSort(input, collation, offset, fetch);
		}
	}

	/**
	 * Implementation of {@link RelFactories.SortFactory} that
	 * returns a vanilla {@link Sort}.
	 */
	private static class SortFactoryImpl implements SortFactory {
		@Override
		public RelNode createSort(RelNode input, RelCollation collation,
				RexNode offset, RexNode fetch) {
			return LogicalSort.create(input, collation, offset, fetch);
		}
	}

	/**
	 * Can create a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Exchange}
	 * of the appropriate type for a rule's calling convention.
	 */
	public interface ExchangeFactory {
		/** Creates a Exchange. */
		RelNode createExchange(RelNode input, RelDistribution distribution);
	}

	/**
	 * Implementation of
	 * {@link RelFactories.ExchangeFactory}
	 * that returns a {@link Exchange}.
	 */
	private static class ExchangeFactoryImpl implements ExchangeFactory {
		@Override public RelNode createExchange(
				RelNode input, RelDistribution distribution) {
			return LogicalExchange.create(input, distribution);
		}
	}

	/**
	 * Can create a {@link SortExchange}
	 * of the appropriate type for a rule's calling convention.
	 */
	public interface SortExchangeFactory {
		/**
		 * Creates a {@link SortExchange}.
		 */
		RelNode createSortExchange(
				RelNode input,
				RelDistribution distribution,
				RelCollation collation);
	}

	/**
	 * Implementation of
	 * {@link RelFactories.SortExchangeFactory}
	 * that returns a {@link SortExchange}.
	 */
	private static class SortExchangeFactoryImpl implements SortExchangeFactory {
		@Override public RelNode createSortExchange(
				RelNode input,
				RelDistribution distribution,
				RelCollation collation) {
			return LogicalSortExchange.create(input, distribution, collation);
		}
	}

	/**
	 * Can create a {@link SetOp} for a particular kind of
	 * set operation (UNION, EXCEPT, INTERSECT) and of the appropriate type
	 * for this rule's calling convention.
	 */
	public interface SetOpFactory {
		/** Creates a set operation. */
		RelNode createSetOp(SqlKind kind, List<RelNode> inputs, boolean all);
	}

	/**
	 * Implementation of {@link RelFactories.SetOpFactory} that
	 * returns a vanilla {@link SetOp} for the particular kind of set
	 * operation (UNION, EXCEPT, INTERSECT).
	 */
	private static class SetOpFactoryImpl implements SetOpFactory {
		@Override
		public RelNode createSetOp(SqlKind kind, List<RelNode> inputs,
				boolean all) {
			switch (kind) {
			case UNION:
				return LogicalUnion.create(inputs, all);
			case EXCEPT:
				return LogicalMinus.create(inputs, all);
			case INTERSECT:
				return LogicalIntersect.create(inputs, all);
			default:
				throw new AssertionError("not a set op: " + kind);
			}
		}
	}

	/**
	 * Can create a {@link LogicalAggregate} of the appropriate type
	 * for this rule's calling convention.
	 */
	public interface AggregateFactory {
		/** Creates an aggregate. */
		RelNode createAggregate(RelNode input, List<RelHint> hints, ImmutableBitSet groupSet,
				ImmutableList<ImmutableBitSet> groupSets, List<AggregateCall> aggCalls);

		@Deprecated // to be removed before 1.23
		default RelNode createAggregate(RelNode input, ImmutableBitSet groupSet,
				ImmutableList<ImmutableBitSet> groupSets, List<AggregateCall> aggCalls) {
			return createAggregate(input, ImmutableList.of(), groupSet, groupSets, aggCalls);
		}

		@Deprecated // to be removed before 1.23
		default RelNode createAggregate(RelNode input, boolean indicator,
				ImmutableBitSet groupSet, ImmutableList<ImmutableBitSet> groupSets,
				List<AggregateCall> aggCalls) {
			Aggregate.checkIndicator(indicator);
			return createAggregate(input, ImmutableList.of(), groupSet, groupSets, aggCalls);
		}
	}

	/**
	 * Implementation of {@link RelFactories.AggregateFactory}
	 * that returns a vanilla {@link LogicalAggregate}.
	 */
	private static class AggregateFactoryImpl implements AggregateFactory {
		@Override
		public RelNode createAggregate(RelNode input, List<RelHint> hints,
				ImmutableBitSet groupSet, ImmutableList<ImmutableBitSet> groupSets,
				List<AggregateCall> aggCalls) {
			return LogicalAggregate.create(input, hints, groupSet, groupSets, aggCalls);
		}
	}

	/**
	 * Can create a {@link Filter} of the appropriate type
	 * for this rule's calling convention.
	 */
	public interface FilterFactory {
		/** Creates a filter.
		 *
		 * <p>Some implementations of {@code Filter} do not support correlation
		 * variables, and for these, this method will throw if {@code variablesSet}
		 * is not empty.
		 *
		 * @param input Input relational expression
		 * @param condition Filter condition; only rows for which this condition
		 *   evaluates to TRUE will be emitted
		 * @param variablesSet Correlating variables that are set when reading
		 *   a row from the input, and which may be referenced from inside the
		 *   condition
		 */
		RelNode createFilter(RelNode input, RexNode condition,
				Set<CorrelationId> variablesSet);

		@Deprecated // to be removed before 2.0
		default RelNode createFilter(RelNode input, RexNode condition) {
			return createFilter(input, condition, ImmutableSet.of());
		}
	}

	/**
	 * Implementation of {@link RelFactories.FilterFactory} that
	 * returns a vanilla {@link LogicalFilter}.
	 */
	private static class FilterFactoryImpl implements FilterFactory {
		@Override
		public RelNode createFilter(RelNode input, RexNode condition,
				Set<CorrelationId> variablesSet) {
			return LogicalFilter.create(input, condition,
					ImmutableSet.copyOf(variablesSet));
		}
	}

	/**
	 * Can create a join of the appropriate type for a rule's calling convention.
	 *
	 * <p>The result is typically a {@link Join}.
	 */
	public interface JoinFactory {
		/**
		 * Creates a join.
		 *
		 * @param left             Left input
		 * @param right            Right input
		 * @param hints            Hints
		 * @param condition        Join condition
		 * @param variablesSet     Set of variables that are set by the
		 *                         LHS and used by the RHS and are not available to
		 *                         nodes above this LogicalJoin in the tree
		 * @param joinType         Join type
		 * @param semiJoinDone     Whether this join has been translated to a
		 *                         semi-join
		 */
		RelNode createJoin(RelNode left, RelNode right, List<RelHint> hints,
				RexNode condition, Set<CorrelationId> variablesSet, JoinRelType joinType,
				boolean semiJoinDone);

		@Deprecated // to be removed before 1.23
		default RelNode createJoin(RelNode left, RelNode right, RexNode condition,
				Set<CorrelationId> variablesSet, JoinRelType joinType,
				boolean semiJoinDone) {
			return createJoin(left, right, ImmutableList.of(), condition, variablesSet,
					joinType, semiJoinDone);
		}

		@Deprecated // to be removed before 1.23
		default RelNode createJoin(RelNode left, RelNode right, RexNode condition,
				JoinRelType joinType, Set<String> variablesStopped,
				boolean semiJoinDone) {
			return createJoin(left, right, ImmutableList.of(), condition,
					CorrelationId.setOf(variablesStopped), joinType, semiJoinDone);
		}
	}

	/**
	 * Implementation of {@link JoinFactory} that returns a vanilla
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalJoin}.
	 */
	private static class JoinFactoryImpl implements JoinFactory {
		@Override
		public RelNode createJoin(RelNode left, RelNode right, List<RelHint> hints,
				RexNode condition, Set<CorrelationId> variablesSet,
				JoinRelType joinType, boolean semiJoinDone) {
			return LogicalJoin.create(left, right, hints, condition, variablesSet, joinType,
					semiJoinDone, ImmutableList.of());
		}
	}

	/**
	 * Can create a correlate of the appropriate type for a rule's calling
	 * convention.
	 *
	 * <p>The result is typically a {@link Correlate}.
	 */
	public interface CorrelateFactory {
		/**
		 * Creates a correlate.
		 *
		 * @param left             Left input
		 * @param right            Right input
		 * @param correlationId    Variable name for the row of left input
		 * @param requiredColumns  Required columns
		 * @param joinType         Join type
		 */
		RelNode createCorrelate(RelNode left, RelNode right,
				CorrelationId correlationId, ImmutableBitSet requiredColumns,
				JoinRelType joinType);
	}

	/**
	 * Implementation of {@link CorrelateFactory} that returns a vanilla
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalCorrelate}.
	 */
	private static class CorrelateFactoryImpl implements CorrelateFactory {
		@Override
		public RelNode createCorrelate(RelNode left, RelNode right,
				CorrelationId correlationId, ImmutableBitSet requiredColumns,
				JoinRelType joinType) {
			return LogicalCorrelate.create(left, right, correlationId,
					requiredColumns, joinType);
		}
	}

	/**
	 * Can create a semi-join of the appropriate type for a rule's calling
	 * convention.
	 *
	 * @deprecated Use {@link JoinFactory} instead.
	 */
	@Deprecated // to be removed before 2.0
	public interface SemiJoinFactory {
		/**
		 * Creates a semi-join.
		 *
		 * @param left             Left input
		 * @param right            Right input
		 * @param condition        Join condition
		 */
		RelNode createSemiJoin(RelNode left, RelNode right, RexNode condition);
	}

	/**
	 * Implementation of {@link SemiJoinFactory} that returns a vanilla
	 * {@link Join} with join type as {@link JoinRelType#SEMI}.
	 *
	 * @deprecated Use {@link JoinFactoryImpl} instead.
	 */
	@Deprecated  // to be removed before 2.0
	private static class SemiJoinFactoryImpl implements SemiJoinFactory {
		@Override
		public RelNode createSemiJoin(RelNode left, RelNode right,
				RexNode condition) {
			return LogicalJoin.create(left, right, condition, ImmutableSet.of(), JoinRelType.SEMI,
					false, ImmutableList.of());
		}
	}

	/**
	 * Can create a {@link Values} of the appropriate type for a rule's calling
	 * convention.
	 */
	public interface ValuesFactory {
		/**
		 * Creates a Values.
		 */
		RelNode createValues(RelOptCluster cluster, RelDataType rowType,
				List<ImmutableList<RexLiteral>> tuples);
	}

	/**
	 * Implementation of {@link ValuesFactory} that returns a
	 * {@link LogicalValues}.
	 */
	private static class ValuesFactoryImpl implements ValuesFactory {
		@Override
		public RelNode createValues(RelOptCluster cluster, RelDataType rowType,
				List<ImmutableList<RexLiteral>> tuples) {
			return LogicalValues.create(cluster, rowType,
					ImmutableList.copyOf(tuples));
		}
	}

	/**
	 * Can create a {@link TableScan} of the appropriate type for a rule's calling
	 * convention.
	 */
	public interface TableScanFactory {
		/**
		 * Creates a {@link TableScan}.
		 */
		RelNode createScan(RelOptTable.ToRelContext toRelContext, RelOptTable table);

		@Deprecated // to be removed before 1.23
		default RelNode createScan(RelOptCluster cluster, RelOptTable table) {
			return createScan(ViewExpanders.simpleContext(cluster), table);
		}
	}

	/**
	 * Implementation of {@link TableScanFactory} that returns a
	 * {@link LogicalTableScan}.
	 */
	private static class TableScanFactoryImpl implements TableScanFactory {
		@Override
		public RelNode createScan(RelOptTable.ToRelContext toRelContext, RelOptTable table) {
			return table.toRel(toRelContext);
		}
	}

	/**
	 * Creates a {@link TableScanFactory} that uses a
	 * {@link org.imsi.queryEREngine.apache.calcite.plan.RelOptTable.ViewExpander} to handle
	 * {@link TranslatableTable} instances, and falls back to a default
	 * factory for other tables.
	 *
	 * @param viewExpander View expander
	 * @param tableScanFactory Factory for non-translatable tables
	 * @return Table scan factory
	 *
	 * @deprecated Use the custom context {@code Contexts.of(viewExpander) } for RelBuilder.
	 *
	 */
	@Deprecated // to be removed before 1.23
	@Nonnull public static TableScanFactory expandingScanFactory(
			@Nonnull RelOptTable.ViewExpander viewExpander,
			@Nonnull TableScanFactory tableScanFactory) {
		return (toRelContext, table) -> {
			final TranslatableTable translatableTable =
					table.unwrap(TranslatableTable.class);
			final RelOptTable.ToRelContext newToRelContext =
					ViewExpanders.toRelContext(
							viewExpander,
							toRelContext.getCluster(),
							toRelContext.getTableHints());
			if (translatableTable != null) {
				return translatableTable.toRel(newToRelContext, table);
			}
			return tableScanFactory.createScan(newToRelContext, table);
		};
	}

	/**
	 * Can create a {@link TableFunctionScan}
	 * of the appropriate type for a rule's calling convention.
	 */
	public interface TableFunctionScanFactory {
		/** Creates a {@link TableFunctionScan}. */
		RelNode createTableFunctionScan(RelOptCluster cluster,
				List<RelNode> inputs, RexNode rexCall, Type elementType,
				Set<RelColumnMapping> columnMappings);
	}

	/**
	 * Implementation of
	 * {@link TableFunctionScanFactory}
	 * that returns a {@link TableFunctionScan}.
	 */
	private static class TableFunctionScanFactoryImpl
	implements TableFunctionScanFactory {
		@Override public RelNode createTableFunctionScan(RelOptCluster cluster,
				List<RelNode> inputs, RexNode rexCall, Type elementType,
				Set<RelColumnMapping> columnMappings) {
			return LogicalTableFunctionScan.create(cluster, inputs, rexCall,
					elementType, rexCall.getType(), columnMappings);
		}
	}

	/**
	 * Can create a {@link Snapshot} of
	 * the appropriate type for a rule's calling convention.
	 */
	public interface SnapshotFactory {
		/**
		 * Creates a {@link Snapshot}.
		 */
		RelNode createSnapshot(RelNode input, RexNode period);
	}

	/**
	 * Implementation of {@link RelFactories.SnapshotFactory} that
	 * returns a vanilla {@link LogicalSnapshot}.
	 */
	public static class SnapshotFactoryImpl implements SnapshotFactory {
		@Override
		public RelNode createSnapshot(RelNode input, RexNode period) {
			return LogicalSnapshot.create(input, period);
		}
	}

	/**
	 * Can create a {@link Match} of
	 * the appropriate type for a rule's calling convention.
	 */
	public interface MatchFactory {
		/** Creates a {@link Match}. */
		RelNode createMatch(RelNode input, RexNode pattern,
				RelDataType rowType, boolean strictStart, boolean strictEnd,
				Map<String, RexNode> patternDefinitions, Map<String, RexNode> measures,
				RexNode after, Map<String, ? extends SortedSet<String>> subsets,
						boolean allRows, ImmutableBitSet partitionKeys, RelCollation orderKeys,
						RexNode interval);
	}

	/**
	 * Implementation of {@link MatchFactory}
	 * that returns a {@link LogicalMatch}.
	 */
	private static class MatchFactoryImpl implements MatchFactory {
		@Override
		public RelNode createMatch(RelNode input, RexNode pattern,
				RelDataType rowType, boolean strictStart, boolean strictEnd,
				Map<String, RexNode> patternDefinitions, Map<String, RexNode> measures,
				RexNode after, Map<String, ? extends SortedSet<String>> subsets,
						boolean allRows, ImmutableBitSet partitionKeys, RelCollation orderKeys,
						RexNode interval) {
			return LogicalMatch.create(input, rowType, pattern, strictStart,
					strictEnd, patternDefinitions, measures, after, subsets, allRows,
					partitionKeys, orderKeys, interval);
		}
	}

	/**
	 * Can create a {@link Spool} of
	 * the appropriate type for a rule's calling convention.
	 */
	@Experimental
	public interface SpoolFactory {
		/** Creates a {@link TableSpool}. */
		RelNode createTableSpool(RelNode input, Spool.Type readType,
				Spool.Type writeType, RelOptTable table);
	}

	/**
	 * Implementation of {@link SpoolFactory}
	 * that returns Logical Spools.
	 */
	private static class SpoolFactoryImpl implements SpoolFactory {
		@Override
		public RelNode createTableSpool(RelNode input, Spool.Type readType,
				Spool.Type writeType, RelOptTable table) {
			return LogicalTableSpool.create(input, readType, writeType, table);
		}
	}

	/**
	 * Can create a {@link RepeatUnion} of
	 * the appropriate type for a rule's calling convention.
	 */
	@Experimental
	public interface RepeatUnionFactory {
		/** Creates a {@link RepeatUnion}. */
		RelNode createRepeatUnion(RelNode seed, RelNode iterative, boolean all,
				int iterationLimit);
	}

	/**
	 * Implementation of {@link RepeatUnion}
	 * that returns a {@link LogicalRepeatUnion}.
	 */
	private static class RepeatUnionFactoryImpl implements RepeatUnionFactory {
		@Override
		public RelNode createRepeatUnion(RelNode seed, RelNode iterative,
				boolean all, int iterationLimit) {
			return LogicalRepeatUnion.create(seed, iterative, all, iterationLimit);
		}
	}

	/** Immutable record that contains an instance of each factory. */
	public static class Struct {
		public final FilterFactory filterFactory;
		public final ProjectFactory projectFactory;
		public final AggregateFactory aggregateFactory;
		public final SortFactory sortFactory;
		public final ExchangeFactory exchangeFactory;
		public final SortExchangeFactory sortExchangeFactory;
		public final SetOpFactory setOpFactory;
		public final JoinFactory joinFactory;
		public final CorrelateFactory correlateFactory;
		public final ValuesFactory valuesFactory;
		public final TableScanFactory scanFactory;
		public final TableFunctionScanFactory tableFunctionScanFactory;
		public final SnapshotFactory snapshotFactory;
		public final MatchFactory matchFactory;
		public final SpoolFactory spoolFactory;
		public final RepeatUnionFactory repeatUnionFactory;

		private Struct(FilterFactory filterFactory,
				ProjectFactory projectFactory,
				AggregateFactory aggregateFactory,
				SortFactory sortFactory,
				ExchangeFactory exchangeFactory,
				SortExchangeFactory sortExchangeFactory,
				SetOpFactory setOpFactory,
				JoinFactory joinFactory,
				CorrelateFactory correlateFactory,
				ValuesFactory valuesFactory,
				TableScanFactory scanFactory,
				TableFunctionScanFactory tableFunctionScanFactory,
				SnapshotFactory snapshotFactory,
				MatchFactory matchFactory,
				SpoolFactory spoolFactory,
				RepeatUnionFactory repeatUnionFactory) {
			this.filterFactory = Objects.requireNonNull(filterFactory);
			this.projectFactory = Objects.requireNonNull(projectFactory);
			this.aggregateFactory = Objects.requireNonNull(aggregateFactory);
			this.sortFactory = Objects.requireNonNull(sortFactory);
			this.exchangeFactory = Objects.requireNonNull(exchangeFactory);
			this.sortExchangeFactory = Objects.requireNonNull(sortExchangeFactory);
			this.setOpFactory = Objects.requireNonNull(setOpFactory);
			this.joinFactory = Objects.requireNonNull(joinFactory);
			this.correlateFactory = Objects.requireNonNull(correlateFactory);
			this.valuesFactory = Objects.requireNonNull(valuesFactory);
			this.scanFactory = Objects.requireNonNull(scanFactory);
			this.tableFunctionScanFactory =
					Objects.requireNonNull(tableFunctionScanFactory);
			this.snapshotFactory = Objects.requireNonNull(snapshotFactory);
			this.matchFactory = Objects.requireNonNull(matchFactory);
			this.spoolFactory = Objects.requireNonNull(spoolFactory);
			this.repeatUnionFactory = Objects.requireNonNull(repeatUnionFactory);
		}

		public static @Nonnull Struct fromContext(Context context) {
			Struct struct = context.unwrap(Struct.class);
			if (struct != null) {
				return struct;
			}
			return new Struct(
					Util.first(context.unwrap(FilterFactory.class),
							DEFAULT_FILTER_FACTORY),
					Util.first(context.unwrap(ProjectFactory.class),
							DEFAULT_PROJECT_FACTORY),
					Util.first(context.unwrap(AggregateFactory.class),
							DEFAULT_AGGREGATE_FACTORY),
					Util.first(context.unwrap(SortFactory.class),
							DEFAULT_SORT_FACTORY),
					Util.first(context.unwrap(ExchangeFactory.class),
							DEFAULT_EXCHANGE_FACTORY),
					Util.first(context.unwrap(SortExchangeFactory.class),
							DEFAULT_SORT_EXCHANGE_FACTORY),
					Util.first(context.unwrap(SetOpFactory.class),
							DEFAULT_SET_OP_FACTORY),
					Util.first(context.unwrap(JoinFactory.class),
							DEFAULT_JOIN_FACTORY),
					Util.first(context.unwrap(CorrelateFactory.class),
							DEFAULT_CORRELATE_FACTORY),
					Util.first(context.unwrap(ValuesFactory.class),
							DEFAULT_VALUES_FACTORY),
					Util.first(context.unwrap(TableScanFactory.class),
							DEFAULT_TABLE_SCAN_FACTORY),
					Util.first(context.unwrap(TableFunctionScanFactory.class),
							DEFAULT_TABLE_FUNCTION_SCAN_FACTORY),
					Util.first(context.unwrap(SnapshotFactory.class),
							DEFAULT_SNAPSHOT_FACTORY),
					Util.first(context.unwrap(MatchFactory.class),
							DEFAULT_MATCH_FACTORY),
					Util.first(context.unwrap(SpoolFactory.class),
							DEFAULT_SPOOL_FACTORY),
					Util.first(context.unwrap(RepeatUnionFactory.class),
							DEFAULT_REPEAT_UNION_FACTORY));
		}
	}
}
