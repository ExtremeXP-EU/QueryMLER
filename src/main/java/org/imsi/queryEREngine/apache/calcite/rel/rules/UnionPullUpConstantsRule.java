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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptPredicateList;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.core.Union;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexInputRef;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableBitSet;
import org.imsi.queryEREngine.apache.calcite.util.Pair;
import org.imsi.queryEREngine.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableList;

/**
 * Planner rule that pulls up constants through a Union operator.
 */
public class UnionPullUpConstantsRule extends RelOptRule {

	public static final UnionPullUpConstantsRule INSTANCE =
			new UnionPullUpConstantsRule(Union.class, RelFactories.LOGICAL_BUILDER);

	/** Creates a UnionPullUpConstantsRule. */
	public UnionPullUpConstantsRule(Class<? extends Union> unionClass,
			RelBuilderFactory relBuilderFactory) {
		// If field count is 1, then there's no room for
		// optimization since we cannot create an empty Project
		// operator. If we created a Project with one column, this rule would
		// cycle.
		super(
				operandJ(unionClass, null, union -> union.getRowType().getFieldCount() > 1, any()),
				relBuilderFactory, null);
	}

	@Override public void onMatch(RelOptRuleCall call) {
		final Union union = call.rel(0);

		final RexBuilder rexBuilder = union.getCluster().getRexBuilder();
		final RelMetadataQuery mq = call.getMetadataQuery();
		final RelOptPredicateList predicates = mq.getPulledUpPredicates(union);
		if (predicates == null) {
			return;
		}

		final Map<Integer, RexNode> constants = new HashMap<>();
		for (Map.Entry<RexNode, RexNode> e : predicates.constantMap.entrySet()) {
			if (e.getKey() instanceof RexInputRef) {
				constants.put(((RexInputRef) e.getKey()).getIndex(), e.getValue());
			}
		}

		// None of the expressions are constant. Nothing to do.
		if (constants.isEmpty()) {
			return;
		}

		// Create expressions for Project operators before and after the Union
		List<RelDataTypeField> fields = union.getRowType().getFieldList();
		List<RexNode> topChildExprs = new ArrayList<>();
		List<String> topChildExprsFields = new ArrayList<>();
		List<RexNode> refs = new ArrayList<>();
		ImmutableBitSet.Builder refsIndexBuilder = ImmutableBitSet.builder();
		for (RelDataTypeField field : fields) {
			final RexNode constant = constants.get(field.getIndex());
			if (constant != null) {
				topChildExprs.add(constant);
				topChildExprsFields.add(field.getName());
			} else {
				final RexNode expr = rexBuilder.makeInputRef(union, field.getIndex());
				topChildExprs.add(expr);
				topChildExprsFields.add(field.getName());
				refs.add(expr);
				refsIndexBuilder.set(field.getIndex());
			}
		}
		ImmutableBitSet refsIndex = refsIndexBuilder.build();

		// Update top Project positions
		final Mappings.TargetMapping mapping =
				RelOptUtil.permutation(refs, union.getInput(0).getRowType()).inverse();
		topChildExprs = ImmutableList.copyOf(RexUtil.apply(mapping, topChildExprs));

		// Create new Project-Union-Project sequences
		final RelBuilder relBuilder = call.builder();
		for (RelNode input : union.getInputs()) {
			List<Pair<RexNode, String>> newChildExprs = new ArrayList<>();
			for (int j : refsIndex) {
				newChildExprs.add(
						Pair.of(rexBuilder.makeInputRef(input, j),
								input.getRowType().getFieldList().get(j).getName()));
			}
			if (newChildExprs.isEmpty()) {
				// At least a single item in project is required.
				newChildExprs.add(
						Pair.of(topChildExprs.get(0), topChildExprsFields.get(0)));
			}
			// Add the input with project on top
			relBuilder.push(input);
			relBuilder.project(Pair.left(newChildExprs), Pair.right(newChildExprs));
		}
		relBuilder.union(union.all, union.getInputs().size());
		// Create top Project fixing nullability of fields
		relBuilder.project(topChildExprs, topChildExprsFields);
		relBuilder.convert(union.getRowType(), false);

		call.transformTo(relBuilder.build());
	}

}
