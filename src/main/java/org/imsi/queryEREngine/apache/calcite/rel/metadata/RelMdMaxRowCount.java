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
package org.imsi.queryEREngine.apache.calcite.rel.metadata;

import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.EnumerableLimit;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPredicateList;
import org.imsi.queryEREngine.apache.calcite.plan.volcano.RelSubset;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Aggregate;
import org.imsi.queryEREngine.apache.calcite.rel.core.Exchange;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.Intersect;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.Minus;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.Sort;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableModify;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableScan;
import org.imsi.queryEREngine.apache.calcite.rel.core.Union;
import org.imsi.queryEREngine.apache.calcite.rel.core.Values;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexLiteral;
import org.imsi.queryEREngine.apache.calcite.util.Bug;
import org.imsi.queryEREngine.apache.calcite.util.BuiltInMethod;
import org.imsi.queryEREngine.apache.calcite.util.Util;

/**
 * RelMdMaxRowCount supplies a default implementation of
 * {@link RelMetadataQuery#getMaxRowCount} for the standard logical algebra.
 */
public class RelMdMaxRowCount
implements MetadataHandler<BuiltInMetadata.MaxRowCount> {
	public static final RelMetadataProvider SOURCE =
			ReflectiveRelMetadataProvider.reflectiveSource(
					BuiltInMethod.MAX_ROW_COUNT.method, new RelMdMaxRowCount());

	//~ Methods ----------------------------------------------------------------

	@Override
	public MetadataDef<BuiltInMetadata.MaxRowCount> getDef() {
		return BuiltInMetadata.MaxRowCount.DEF;
	}

	public Double getMaxRowCount(Union rel, RelMetadataQuery mq) {
		double rowCount = 0.0;
		for (RelNode input : rel.getInputs()) {
			Double partialRowCount = mq.getMaxRowCount(input);
			if (partialRowCount == null) {
				return null;
			}
			rowCount += partialRowCount;
		}
		return rowCount;
	}

	public Double getMaxRowCount(Intersect rel, RelMetadataQuery mq) {
		// max row count is the smallest of the inputs
		Double rowCount = null;
		for (RelNode input : rel.getInputs()) {
			Double partialRowCount = mq.getMaxRowCount(input);
			if (rowCount == null
					|| partialRowCount != null && partialRowCount < rowCount) {
				rowCount = partialRowCount;
			}
		}
		return rowCount;
	}

	public Double getMaxRowCount(Minus rel, RelMetadataQuery mq) {
		return mq.getMaxRowCount(rel.getInput(0));
	}

	public Double getMaxRowCount(Filter rel, RelMetadataQuery mq) {
		if (rel.getCondition().isAlwaysFalse()) {
			return 0D;
		}
		return mq.getMaxRowCount(rel.getInput());
	}

	public Double getMaxRowCount(Project rel, RelMetadataQuery mq) {
		return mq.getMaxRowCount(rel.getInput());
	}

	public Double getMaxRowCount(Exchange rel, RelMetadataQuery mq) {
		return mq.getMaxRowCount(rel.getInput());
	}

	public Double getMaxRowCount(Sort rel, RelMetadataQuery mq) {
		Double rowCount = mq.getMaxRowCount(rel.getInput());
		if (rowCount == null) {
			rowCount = Double.POSITIVE_INFINITY;
		}
		final int offset = rel.offset == null ? 0 : RexLiteral.intValue(rel.offset);
		rowCount = Math.max(rowCount - offset, 0D);

		if (rel.fetch != null) {
			final int limit = RexLiteral.intValue(rel.fetch);
			if (limit < rowCount) {
				return (double) limit;
			}
		}
		return rowCount;
	}

	public Double getMaxRowCount(EnumerableLimit rel, RelMetadataQuery mq) {
		Double rowCount = mq.getMaxRowCount(rel.getInput());
		if (rowCount == null) {
			rowCount = Double.POSITIVE_INFINITY;
		}
		final int offset = rel.offset == null ? 0 : RexLiteral.intValue(rel.offset);
		rowCount = Math.max(rowCount - offset, 0D);

		if (rel.fetch != null) {
			final int limit = RexLiteral.intValue(rel.fetch);
			if (limit < rowCount) {
				return (double) limit;
			}
		}
		return rowCount;
	}

	public Double getMaxRowCount(Aggregate rel, RelMetadataQuery mq) {
		if (rel.getGroupSet().isEmpty()) {
			// Aggregate with no GROUP BY always returns 1 row (even on empty table).
			return 1D;
		}

		// Aggregate with constant GROUP BY always returns 1 row
		if (rel.getGroupType() == Aggregate.Group.SIMPLE) {
			final RelOptPredicateList predicateList =
					mq.getPulledUpPredicates(rel.getInput());
			if (predicateList != null
					&& allGroupKeysAreConstant(rel, predicateList)) {
				return 1D;
			}
		}
		final Double rowCount = mq.getMaxRowCount(rel.getInput());
		if (rowCount == null) {
			return null;
		}
		return rowCount * rel.getGroupSets().size();
	}

	private static boolean allGroupKeysAreConstant(Aggregate aggregate,
			RelOptPredicateList predicateList) {
		final RexBuilder rexBuilder = aggregate.getCluster().getRexBuilder();
		for (int key : aggregate.getGroupSet()) {
			if (!predicateList.constantMap.containsKey(
					rexBuilder.makeInputRef(aggregate.getInput(), key))) {
				return false;
			}
		}
		return true;
	}

	public Double getMaxRowCount(Join rel, RelMetadataQuery mq) {
		Double left = mq.getMaxRowCount(rel.getLeft());
		Double right = mq.getMaxRowCount(rel.getRight());
		if (left == null || right == null) {
			return null;
		}
		if (left < 1D && rel.getJoinType().generatesNullsOnLeft()) {
			left = 1D;
		}
		if (right < 1D && rel.getJoinType().generatesNullsOnRight()) {
			right = 1D;
		}
		return left * right;
	}

	public Double getMaxRowCount(TableScan rel, RelMetadataQuery mq) {
		// For typical tables, there is no upper bound to the number of rows.
		return Double.POSITIVE_INFINITY;
	}

	public Double getMaxRowCount(Values values, RelMetadataQuery mq) {
		// For Values, the maximum row count is the actual row count.
		// This is especially useful if Values is empty.
		return (double) values.getTuples().size();
	}

	public Double getMaxRowCount(TableModify rel, RelMetadataQuery mq) {
		return mq.getMaxRowCount(rel.getInput());
	}

	public Double getMaxRowCount(RelSubset rel, RelMetadataQuery mq) {
		// FIXME This is a short-term fix for [CALCITE-1018]. A complete
		// solution will come with [CALCITE-1048].
		Util.discard(Bug.CALCITE_1048_FIXED);
		for (RelNode node : rel.getRels()) {
			if (node instanceof Sort) {
				Sort sort = (Sort) node;
				if (sort.fetch != null) {
					return (double) RexLiteral.intValue(sort.fetch);
				}
			}
		}

		return Double.POSITIVE_INFINITY;
	}

	// Catch-all rule when none of the others apply.
	public Double getMaxRowCount(RelNode rel, RelMetadataQuery mq) {
		return null;
	}
}
