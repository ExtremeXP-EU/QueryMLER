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

import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.JoinRelType;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalFilter;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalJoin;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

/**
 * Planner rule that pushes {@code SemiJoin}s down in a tree past
 * a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Filter}.
 *
 * <p>The intention is to trigger other rules that will convert
 * {@code SemiJoin}s.
 *
 * <p>SemiJoin(LogicalFilter(X), Y) &rarr; LogicalFilter(SemiJoin(X, Y))
 *
 * @see SemiJoinProjectTransposeRule
 */
public class SemiJoinFilterTransposeRule extends RelOptRule {
	public static final SemiJoinFilterTransposeRule INSTANCE =
			new SemiJoinFilterTransposeRule(RelFactories.LOGICAL_BUILDER);

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a SemiJoinFilterTransposeRule.
	 */
	public SemiJoinFilterTransposeRule(RelBuilderFactory relBuilderFactory) {
		super(
				operandJ(LogicalJoin.class, null, Join::isSemiJoin,
						some(operand(LogicalFilter.class, any()))),
				relBuilderFactory, null);
	}

	//~ Methods ----------------------------------------------------------------

	// implement RelOptRule
	@Override
	public void onMatch(RelOptRuleCall call) {
		LogicalJoin semiJoin = call.rel(0);
		LogicalFilter filter = call.rel(1);

		RelNode newSemiJoin =
				LogicalJoin.create(filter.getInput(),
						semiJoin.getRight(),
						// No need to copy the hints, the framework would try to do that.
						ImmutableList.of(),
						semiJoin.getCondition(),
						ImmutableSet.of(),
						JoinRelType.SEMI);

		final RelFactories.FilterFactory factory =
				RelFactories.DEFAULT_FILTER_FACTORY;
		RelNode newFilter =
				factory.createFilter(newSemiJoin, filter.getCondition(),
						ImmutableSet.of());

		call.transformTo(newFilter);
	}
}
