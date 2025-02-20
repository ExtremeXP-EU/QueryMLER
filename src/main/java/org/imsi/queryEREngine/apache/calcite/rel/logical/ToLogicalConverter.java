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
package org.imsi.queryEREngine.apache.calcite.rel.logical;

import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.EnumerableInterpreter;
import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.EnumerableLimit;
import org.imsi.queryEREngine.apache.calcite.adapter.jdbc.JdbcToEnumerableConverter;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollations;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelShuttleImpl;
import org.imsi.queryEREngine.apache.calcite.rel.SingleRel;
import org.imsi.queryEREngine.apache.calcite.rel.core.Aggregate;
import org.imsi.queryEREngine.apache.calcite.rel.core.Calc;
import org.imsi.queryEREngine.apache.calcite.rel.core.Correlate;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.Intersect;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.Minus;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.Sort;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableModify;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableScan;
import org.imsi.queryEREngine.apache.calcite.rel.core.Uncollect;
import org.imsi.queryEREngine.apache.calcite.rel.core.Union;
import org.imsi.queryEREngine.apache.calcite.rel.core.Values;
import org.imsi.queryEREngine.apache.calcite.rel.core.Window;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableBitSet;

/**
 * Shuttle to convert any rel plan to a plan with all logical nodes.
 */
public class ToLogicalConverter extends RelShuttleImpl {
	private final RelBuilder relBuilder;

	public ToLogicalConverter(RelBuilder relBuilder) {
		this.relBuilder = relBuilder;
	}

	@Override public RelNode visit(TableScan scan) {
		return LogicalTableScan.create(scan.getCluster(), scan.getTable(), scan.getHints());
	}

	@Override public RelNode visit(RelNode relNode) {
		if (relNode instanceof Aggregate) {
			final Aggregate agg = (Aggregate) relNode;
			return relBuilder.push(visit(agg.getInput()))
					.aggregate(
							relBuilder.groupKey(agg.getGroupSet(),
									(Iterable<ImmutableBitSet>) agg.groupSets),
							agg.getAggCallList())
					.build();
		}

		if (relNode instanceof TableScan) {
			return visit((TableScan) relNode);
		}

		if (relNode instanceof Filter) {
			final Filter filter = (Filter) relNode;
			return relBuilder.push(visit(filter.getInput()))
					.filter(filter.getCondition())
					.build();
		}

		if (relNode instanceof Project) {
			final Project project = (Project) relNode;
			return relBuilder.push(visit(project.getInput()))
					.project(project.getProjects(), project.getRowType().getFieldNames())
					.build();
		}

		if (relNode instanceof Union) {
			final Union union = (Union) relNode;
			for (RelNode rel : union.getInputs()) {
				relBuilder.push(visit(rel));
			}
			return relBuilder.union(union.all, union.getInputs().size())
					.build();
		}

		if (relNode instanceof Intersect) {
			final Intersect intersect = (Intersect) relNode;
			for (RelNode rel : intersect.getInputs()) {
				relBuilder.push(visit(rel));
			}
			return relBuilder.intersect(intersect.all, intersect.getInputs().size())
					.build();
		}

		if (relNode instanceof Minus) {
			final Minus minus = (Minus) relNode;
			for (RelNode rel : minus.getInputs()) {
				relBuilder.push(visit(rel));
			}
			return relBuilder.minus(minus.all, minus.getInputs().size())
					.build();
		}

		if (relNode instanceof Join) {
			final Join join = (Join) relNode;
			return relBuilder.push(visit(join.getLeft()))
					.push(visit(join.getRight()))
					.join(join.getJoinType(), join.getCondition())
					.build();
		}

		if (relNode instanceof Correlate) {
			final Correlate corr = (Correlate) relNode;
			return relBuilder.push(visit(corr.getLeft()))
					.push(visit(corr.getRight()))
					.join(corr.getJoinType(), relBuilder.literal(true),
							corr.getVariablesSet())
					.build();
		}

		if (relNode instanceof Values) {
			final Values values = (Values) relNode;
			return relBuilder.values(values.tuples, values.getRowType())
					.build();
		}

		if (relNode instanceof Sort) {
			final Sort sort = (Sort) relNode;
			return LogicalSort.create(visit(sort.getInput()), sort.getCollation(),
					sort.offset, sort.fetch);
		}

		if (relNode instanceof Window) {
			final Window window = (Window) relNode;
			final RelNode input = visit(window.getInput());
			return LogicalWindow.create(input.getTraitSet(), input, window.constants,
					window.getRowType(), window.groups);
		}

		if (relNode instanceof Calc) {
			final Calc calc = (Calc) relNode;
			return LogicalCalc.create(visit(calc.getInput()), calc.getProgram());
		}

		if (relNode instanceof TableModify) {
			final TableModify tableModify = (TableModify) relNode;
			final RelNode input = visit(tableModify.getInput());
			return LogicalTableModify.create(tableModify.getTable(),
					tableModify.getCatalogReader(), input, tableModify.getOperation(),
					tableModify.getUpdateColumnList(), tableModify.getSourceExpressionList(),
					tableModify.isFlattened());
		}

		if (relNode instanceof EnumerableInterpreter
				|| relNode instanceof JdbcToEnumerableConverter) {
			return visit(((SingleRel) relNode).getInput());
		}

		if (relNode instanceof EnumerableLimit) {
			final EnumerableLimit limit = (EnumerableLimit) relNode;
			RelNode logicalInput = visit(limit.getInput());
			RelCollation collation = RelCollations.of();
			if (logicalInput instanceof Sort) {
				collation = ((Sort) logicalInput).collation;
				logicalInput = ((Sort) logicalInput).getInput();
			}
			return LogicalSort.create(logicalInput, collation, limit.offset,
					limit.fetch);
		}

		if (relNode instanceof Uncollect) {
			final Uncollect uncollect = (Uncollect) relNode;
			final RelNode input = visit(uncollect.getInput());
			return new Uncollect(input.getCluster(), input.getTraitSet(), input,
					uncollect.withOrdinality);
		}

		throw new AssertionError("Need to implement logical converter for "
				+ relNode.getClass().getName());
	}
}
