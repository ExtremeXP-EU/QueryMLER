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

import java.util.ArrayList;
import java.util.List;

import org.imsi.queryEREngine.apache.calcite.plan.Contexts;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.core.SetOp;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/**
 * Planner rule that pushes a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Filter}
 * past a {@link org.imsi.queryEREngine.apache.calcite.rel.core.SetOp}.
 */
public class FilterSetOpTransposeRule extends RelOptRule {
	public static final FilterSetOpTransposeRule INSTANCE =
			new FilterSetOpTransposeRule(RelFactories.LOGICAL_BUILDER);

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a FilterSetOpTransposeRule.
	 */
	public FilterSetOpTransposeRule(RelBuilderFactory relBuilderFactory) {
		super(
				operand(Filter.class,
						operand(SetOp.class, any())),
				relBuilderFactory, null);
	}

	@Deprecated // to  be removed before 2.0
	public FilterSetOpTransposeRule(RelFactories.FilterFactory filterFactory) {
		this(RelBuilder.proto(Contexts.of(filterFactory)));
	}

	//~ Methods ----------------------------------------------------------------

	// implement RelOptRule
	@Override
	public void onMatch(RelOptRuleCall call) {
		Filter filterRel = call.rel(0);
		SetOp setOp = call.rel(1);

		RexNode condition = filterRel.getCondition();

		// create filters on top of each setop child, modifying the filter
		// condition to reference each setop child
		RexBuilder rexBuilder = filterRel.getCluster().getRexBuilder();
		final RelBuilder relBuilder = call.builder();
		List<RelDataTypeField> origFields =
				setOp.getRowType().getFieldList();
		int[] adjustments = new int[origFields.size()];
		final List<RelNode> newSetOpInputs = new ArrayList<>();
		for (RelNode input : setOp.getInputs()) {
			RexNode newCondition =
					condition.accept(
							new RelOptUtil.RexInputConverter(
									rexBuilder,
									origFields,
									input.getRowType().getFieldList(),
									adjustments));
			newSetOpInputs.add(relBuilder.push(input).filter(newCondition).build());
		}

		// create a new setop whose children are the filters created above
		SetOp newSetOp =
				setOp.copy(setOp.getTraitSet(), newSetOpInputs);

		call.transformTo(newSetOp);
	}
}
