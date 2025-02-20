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
package org.imsi.queryEREngine.apache.calcite.rel.rules;

import static org.imsi.queryEREngine.apache.calcite.plan.RelOptRule.any;
import static org.imsi.queryEREngine.apache.calcite.plan.RelOptRule.none;
import static org.imsi.queryEREngine.apache.calcite.plan.RelOptRule.operand;
import static org.imsi.queryEREngine.apache.calcite.plan.RelOptRule.operandJ;
import static org.imsi.queryEREngine.apache.calcite.plan.RelOptRule.some;
import static org.imsi.queryEREngine.apache.calcite.plan.RelOptRule.unordered;

import java.util.List;
import java.util.function.Predicate;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.plan.hep.HepRelVertex;
import org.imsi.queryEREngine.apache.calcite.plan.volcano.RelSubset;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.SingleRel;
import org.imsi.queryEREngine.apache.calcite.rel.core.Aggregate;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.JoinRelType;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.core.Sort;
import org.imsi.queryEREngine.apache.calcite.rel.core.Values;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalIntersect;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalMinus;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalUnion;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalValues;
import org.imsi.queryEREngine.apache.calcite.rex.RexDynamicParam;
import org.imsi.queryEREngine.apache.calcite.rex.RexLiteral;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/**
 * Collection of rules which remove sections of a query plan known never to
 * produce any rows.
 *
 * <p>Conventionally, the way to represent an empty relational expression is
 * with a {@link Values} that has no tuples.
 *
 * @see LogicalValues#createEmpty
 */
public abstract class PruneEmptyRules {
	//~ Static fields/initializers ---------------------------------------------

	/**
	 * Rule that removes empty children of a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalUnion}.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Union(Rel, Empty, Rel2) becomes Union(Rel, Rel2)
	 * <li>Union(Rel, Empty, Empty) becomes Rel
	 * <li>Union(Empty, Empty) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule UNION_INSTANCE =
			new RelOptRule(
					operand(LogicalUnion.class,
							unordered(operandJ(Values.class, null, Values::isEmpty, none()))),
					"Union") {
		@Override
		public void onMatch(RelOptRuleCall call) {
			final LogicalUnion union = call.rel(0);
			final List<RelNode> inputs = union.getInputs();
			assert inputs != null;
			final RelBuilder builder = call.builder();
			int nonEmptyInputs = 0;
			for (RelNode input : inputs) {
				if (!isEmpty(input)) {
					builder.push(input);
					nonEmptyInputs++;
				}
			}
			assert nonEmptyInputs < inputs.size()
			: "planner promised us at least one Empty child: " + RelOptUtil.toString(union);
			if (nonEmptyInputs == 0) {
				builder.push(union).empty();
			} else {
				builder.union(union.all, nonEmptyInputs);
				builder.convert(union.getRowType(), true);
			}
			call.transformTo(builder.build());
		}
	};

	/**
	 * Rule that removes empty children of a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalMinus}.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Minus(Rel, Empty, Rel2) becomes Minus(Rel, Rel2)
	 * <li>Minus(Empty, Rel) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule MINUS_INSTANCE =
			new RelOptRule(
					operand(LogicalMinus.class,
							unordered(
									operandJ(Values.class, null, Values::isEmpty, none()))),
					"Minus") {
		@Override
		public void onMatch(RelOptRuleCall call) {
			final LogicalMinus minus = call.rel(0);
			final List<RelNode> inputs = minus.getInputs();
			assert inputs != null;
			int nonEmptyInputs = 0;
			final RelBuilder builder = call.builder();
			for (RelNode input : inputs) {
				if (!isEmpty(input)) {
					builder.push(input);
					nonEmptyInputs++;
				} else if (nonEmptyInputs == 0) {
					// If the first input of Minus is empty, the whole thing is
					// empty.
					break;
				}
			}
			assert nonEmptyInputs < inputs.size()
			: "planner promised us at least one Empty child: " + RelOptUtil.toString(minus);
			if (nonEmptyInputs == 0) {
				builder.push(minus).empty();
			} else {
				builder.minus(minus.all, nonEmptyInputs);
				builder.convert(minus.getRowType(), true);
			}
			call.transformTo(builder.build());
		}
	};

	/**
	 * Rule that converts a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalIntersect} to
	 * empty if any of its children are empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Intersect(Rel, Empty, Rel2) becomes Empty
	 * <li>Intersect(Empty, Rel) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule INTERSECT_INSTANCE =
			new RelOptRule(
					operand(LogicalIntersect.class,
							unordered(
									operandJ(Values.class, null, Values::isEmpty, none()))),
					"Intersect") {
		@Override
		public void onMatch(RelOptRuleCall call) {
			LogicalIntersect intersect = call.rel(0);
			final RelBuilder builder = call.builder();
			builder.push(intersect).empty();
			call.transformTo(builder.build());
		}
	};

	private static boolean isEmpty(RelNode node) {
		if (node instanceof Values) {
			return ((Values) node).getTuples().isEmpty();
		}
		if (node instanceof HepRelVertex) {
			return isEmpty(((HepRelVertex) node).getCurrentRel());
		}
		// Note: relation input might be a RelSubset, so we just iterate over the relations
		// in order to check if the subset is equivalent to an empty relation.
		if (!(node instanceof RelSubset)) {
			return false;
		}
		RelSubset subset = (RelSubset) node;
		for (RelNode rel : subset.getRels()) {
			if (isEmpty(rel)) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Rule that converts a {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject}
	 * to empty if its child is empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Project(Empty) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule PROJECT_INSTANCE =
			new RemoveEmptySingleRule(Project.class,
					(Predicate<Project>) project -> true, RelFactories.LOGICAL_BUILDER,
					"PruneEmptyProject");

	/**
	 * Rule that converts a {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalFilter}
	 * to empty if its child is empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Filter(Empty) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule FILTER_INSTANCE =
			new RemoveEmptySingleRule(Filter.class, "PruneEmptyFilter");

	/**
	 * Rule that converts a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Sort}
	 * to empty if its child is empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Sort(Empty) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule SORT_INSTANCE =
			new RemoveEmptySingleRule(Sort.class, "PruneEmptySort");

	/**
	 * Rule that converts a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Sort}
	 * to empty if it has {@code LIMIT 0}.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Sort(Empty) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule SORT_FETCH_ZERO_INSTANCE =
			new RelOptRule(
					operand(Sort.class, any()), "PruneSortLimit0") {
		@Override public void onMatch(RelOptRuleCall call) {
			Sort sort = call.rel(0);
			if (sort.fetch != null
					&& !(sort.fetch instanceof RexDynamicParam)
					&& RexLiteral.intValue(sort.fetch) == 0) {
				call.transformTo(call.builder().push(sort).empty().build());
			}
		}
	};

	/**
	 * Rule that converts an {@link org.imsi.queryEREngine.apache.calcite.rel.core.Aggregate}
	 * to empty if its child is empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>{@code Aggregate(key: [1, 3], Empty)} &rarr; {@code Empty}
	 *
	 * <li>{@code Aggregate(key: [], Empty)} is unchanged, because an aggregate
	 * without a GROUP BY key always returns 1 row, even over empty input
	 * </ul>
	 *
	 * @see AggregateValuesRule
	 */
	public static final RelOptRule AGGREGATE_INSTANCE =
			new RemoveEmptySingleRule(Aggregate.class,
					(Predicate<Aggregate>) Aggregate::isNotGrandTotal,
					RelFactories.LOGICAL_BUILDER, "PruneEmptyAggregate");

	/**
	 * Rule that converts a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Join}
	 * to empty if its left child is empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Join(Empty, Scan(Dept), INNER) becomes Empty
	 * <li>Join(Empty, Scan(Dept), LEFT) becomes Empty
	 * <li>Join(Empty, Scan(Dept), SEMI) becomes Empty
	 * <li>Join(Empty, Scan(Dept), ANTI) becomes Empty
	 * </ul>
	 */
	public static final RelOptRule JOIN_LEFT_INSTANCE =
			new RelOptRule(
					operand(Join.class,
							some(
									operandJ(Values.class, null, Values::isEmpty, none()),
									operand(RelNode.class, any()))),
					"PruneEmptyJoin(left)") {
		@Override public void onMatch(RelOptRuleCall call) {
			Join join = call.rel(0);
			if (join.getJoinType().generatesNullsOnLeft()) {
				// "select * from emp right join dept" is not necessarily empty if
				// emp is empty
				return;
			}
			call.transformTo(call.builder().push(join).empty().build());
		}
	};

	/**
	 * Rule that converts a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Join}
	 * to empty if its right child is empty.
	 *
	 * <p>Examples:
	 *
	 * <ul>
	 * <li>Join(Scan(Emp), Empty, INNER) becomes Empty
	 * <li>Join(Scan(Emp), Empty, RIGHT) becomes Empty
	 * <li>Join(Scan(Emp), Empty, SEMI) becomes Empty
	 * <li>Join(Scan(Emp), Empty, ANTI) becomes Scan(Emp)
	 * </ul>
	 */
	public static final RelOptRule JOIN_RIGHT_INSTANCE =
			new RelOptRule(
					operand(Join.class,
							some(
									operand(RelNode.class, any()),
									operandJ(Values.class, null, Values::isEmpty, none()))),
					"PruneEmptyJoin(right)") {
		@Override public void onMatch(RelOptRuleCall call) {
			Join join = call.rel(0);
			if (join.getJoinType().generatesNullsOnRight()) {
				// "select * from emp left join dept" is not necessarily empty if
				// dept is empty
				return;
			}
			if (join.getJoinType() == JoinRelType.ANTI) {
				// "select * from emp anti join dept" is not necessarily empty if dept is empty
				if (join.analyzeCondition().isEqui()) {
					// In case of anti (equi) join: Join(X, Empty, ANTI) becomes X
					call.transformTo(join.getLeft());
				}
				return;
			}
			call.transformTo(call.builder().push(join).empty().build());
		}
	};

	/** Planner rule that converts a single-rel (e.g. project, sort, aggregate or
	 * filter) on top of the empty relational expression into empty. */
	public static class RemoveEmptySingleRule extends RelOptRule {
		/** Creates a simple RemoveEmptySingleRule. */
		public <R extends SingleRel> RemoveEmptySingleRule(Class<R> clazz,
				String description) {
			this(clazz, (Predicate<R>) project -> true, RelFactories.LOGICAL_BUILDER,
					description);
		}

		/** Creates a RemoveEmptySingleRule. */
		public <R extends SingleRel> RemoveEmptySingleRule(Class<R> clazz,
				Predicate<R> predicate, RelBuilderFactory relBuilderFactory,
				String description) {
			super(
					operandJ(clazz, null, predicate,
							operandJ(Values.class, null, Values::isEmpty, none())),
					relBuilderFactory, description);
		}

		@SuppressWarnings("Guava")
		@Deprecated // to be removed before 2.0
		public <R extends SingleRel> RemoveEmptySingleRule(Class<R> clazz,
				com.google.common.base.Predicate<R> predicate,
				RelBuilderFactory relBuilderFactory, String description) {
			this(clazz, (Predicate<R>) predicate::apply, relBuilderFactory,
					description);
		}

		@Override
		public void onMatch(RelOptRuleCall call) {
			SingleRel single = call.rel(0);
			call.transformTo(call.builder().push(single).empty().build());
		}
	}
}
