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

import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollationTraitDef;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollations;
import org.imsi.queryEREngine.apache.calcite.rel.RelFieldCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalJoin;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.rex.RexCall;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexOver;
import org.imsi.queryEREngine.apache.calcite.rex.RexShuttle;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;
import org.imsi.queryEREngine.apache.calcite.util.mapping.Mappings;

/**
 * Planner rule that pushes a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Project}
 * past a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Join}
 * by splitting the projection into a projection on top of each child of
 * the join.
 */
public class ProjectJoinTransposeRule extends RelOptRule {
	/**
	 * A instance for ProjectJoinTransposeRule that pushes a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject}
	 * past a {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalJoin}
	 * by splitting the projection into a projection on top of each child of
	 * the join.
	 */
	public static final ProjectJoinTransposeRule INSTANCE =
			new ProjectJoinTransposeRule(
					LogicalProject.class, LogicalJoin.class,
					expr -> !(expr instanceof RexOver),
					RelFactories.LOGICAL_BUILDER);

	//~ Instance fields --------------------------------------------------------

	/**
	 * Condition for expressions that should be preserved in the projection.
	 */
	private final PushProjector.ExprCondition preserveExprCondition;

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a ProjectJoinTransposeRule with an explicit condition.
	 *
	 * @param preserveExprCondition Condition for expressions that should be
	 *                             preserved in the projection
	 */
	public ProjectJoinTransposeRule(
			Class<? extends Project> projectClass,
			Class<? extends Join> joinClass,
			PushProjector.ExprCondition preserveExprCondition,
			RelBuilderFactory relFactory) {
		super(operand(projectClass, operand(joinClass, any())), relFactory, null);
		this.preserveExprCondition = preserveExprCondition;
	}

	//~ Methods ----------------------------------------------------------------

	// implement RelOptRule
	@Override
	public void onMatch(RelOptRuleCall call) {
		Project origProj = call.rel(0);
		final Join join = call.rel(1);

		if (!join.getJoinType().projectsRight()) {
			return; // TODO: support SemiJoin / AntiJoin
		}

		// Normalize the join condition so we don't end up misidentified expanded
		// form of IS NOT DISTINCT FROM as PushProject also visit the filter condition
		// and push down expressions.
		RexNode joinFilter = join.getCondition().accept(new RexShuttle() {
			@Override public RexNode visitCall(RexCall rexCall) {
				final RexNode node = super.visitCall(rexCall);
				if (!(node instanceof RexCall)) {
					return node;
				}
				return RelOptUtil.collapseExpandedIsNotDistinctFromExpr((RexCall) node,
						call.builder().getRexBuilder());
			}
		});

		// locate all fields referenced in the projection and join condition;
		// determine which inputs are referenced in the projection and
		// join condition; if all fields are being referenced and there are no
		// special expressions, no point in proceeding any further
		PushProjector pushProject =
				new PushProjector(
						origProj,
						joinFilter,
						join,
						preserveExprCondition,
						call.builder());
		if (pushProject.locateAllRefs()) {
			return;
		}

		// create left and right projections, projecting only those
		// fields referenced on each side
		RelNode leftProjRel =
				pushProject.createProjectRefsAndExprs(
						join.getLeft(),
						true,
						false);
		RelNode rightProjRel =
				pushProject.createProjectRefsAndExprs(
						join.getRight(),
						true,
						true);

		// convert the join condition to reference the projected columns
		RexNode newJoinFilter = null;
		int[] adjustments = pushProject.getAdjustments();
		if (joinFilter != null) {
			List<RelDataTypeField> projJoinFieldList = new ArrayList<>();
			projJoinFieldList.addAll(
					join.getSystemFieldList());
			projJoinFieldList.addAll(
					leftProjRel.getRowType().getFieldList());
			projJoinFieldList.addAll(
					rightProjRel.getRowType().getFieldList());
			newJoinFilter =
					pushProject.convertRefsAndExprs(
							joinFilter,
							projJoinFieldList,
							adjustments);
		}
		RelTraitSet traits = join.getTraitSet();
		final List<RelCollation> originCollations = traits.getTraits(RelCollationTraitDef.INSTANCE);

		if (originCollations != null && !originCollations.isEmpty()) {
			List<RelCollation> newCollations = new ArrayList<>();
			final int originLeftCnt = join.getLeft().getRowType().getFieldCount();
			final Mappings.TargetMapping leftMapping = RelOptUtil.permutationPushDownProject(
					((Project) leftProjRel).getProjects(), join.getLeft().getRowType(),
					0, 0);
			final Mappings.TargetMapping rightMapping = RelOptUtil.permutationPushDownProject(
					((Project) rightProjRel).getProjects(), join.getRight().getRowType(),
					originLeftCnt, leftProjRel.getRowType().getFieldCount());
			for (RelCollation collation: originCollations) {
				List<RelFieldCollation> fc = new ArrayList<>();
				final List<RelFieldCollation> fieldCollations = collation.getFieldCollations();
				for (RelFieldCollation relFieldCollation: fieldCollations) {
					final int fieldIndex = relFieldCollation.getFieldIndex();
					if (fieldIndex < originLeftCnt) {
						fc.add(RexUtil.apply(leftMapping, relFieldCollation));
					} else {
						fc.add(RexUtil.apply(rightMapping, relFieldCollation));
					}
				}
				newCollations.add(RelCollations.of(fc));
			}
			if (!newCollations.isEmpty()) {
				traits = traits.replace(newCollations);
			}
		}
		// create a new join with the projected children
		Join newJoinRel =
				join.copy(
						traits,
						newJoinFilter,
						leftProjRel,
						rightProjRel,
						join.getJoinType(),
						join.isSemiJoinDone());

		// put the original project on top of the join, converting it to
		// reference the modified projection list
		RelNode topProject =
				pushProject.createNewProject(newJoinRel, adjustments);

		call.transformTo(topProject);
	}
}
