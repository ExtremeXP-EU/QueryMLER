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

import java.util.Collections;
import java.util.function.Predicate;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleOperand;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollationTraitDef;
import org.imsi.queryEREngine.apache.calcite.rel.RelDistributionTraitDef;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexOver;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/**
 * Planner rule that pushes
 * a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Filter}
 * past a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Project}.
 */
public class FilterProjectTransposeRule extends RelOptRule {
	/** The default instance of
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.rules.FilterProjectTransposeRule}.
	 *
	 * <p>It matches any kind of {@link org.imsi.queryEREngine.apache.calcite.rel.core.Join} or
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.core.Filter}, and generates the same kind of
	 * Join and Filter.
	 *
	 * <p>It does not allow a Filter to be pushed past the Project if
	 * {@link RexUtil#containsCorrelation there is a correlation condition})
	 * anywhere in the Filter, since in some cases it can prevent a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.core.Correlate} from being de-correlated.
	 */
	public static final FilterProjectTransposeRule INSTANCE =
			new FilterProjectTransposeRule(Filter.class, Project.class, true, true,
					RelFactories.LOGICAL_BUILDER);

	private final boolean copyFilter;
	private final boolean copyProject;

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a FilterProjectTransposeRule.
	 *
	 * <p>Equivalent to the rule created by
	 * {@link #FilterProjectTransposeRule(Class, Predicate, Class, Predicate, boolean, boolean, RelBuilderFactory)}
	 * with some default predicates that do not allow a filter to be pushed
	 * past the project if there is a correlation condition anywhere in the
	 * filter (since in some cases it can prevent a
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.core.Correlate} from being de-correlated).
	 */
	public FilterProjectTransposeRule(
			Class<? extends Filter> filterClass,
			Class<? extends Project> projectClass,
			boolean copyFilter, boolean copyProject,
			RelBuilderFactory relBuilderFactory) {
		this(filterClass,
				filter -> !RexUtil.containsCorrelation(filter.getCondition()),
				projectClass, project -> true,
				copyFilter, copyProject, relBuilderFactory);
	}

	/**
	 * Creates a FilterProjectTransposeRule.
	 *
	 * <p>If {@code copyFilter} is true, creates the same kind of Filter as
	 * matched in the rule, otherwise it creates a Filter using the RelBuilder
	 * obtained by the {@code relBuilderFactory}.
	 * Similarly for {@code copyProject}.
	 *
	 * <p>Defining predicates for the Filter (using {@code filterPredicate})
	 * and/or the Project (using {@code projectPredicate} allows making the rule
	 * more restrictive.
	 */
	public <F extends Filter, P extends Project> FilterProjectTransposeRule(
			Class<F> filterClass,
			Predicate<? super F> filterPredicate,
			Class<P> projectClass,
			Predicate<? super P> projectPredicate,
			boolean copyFilter, boolean copyProject,
			RelBuilderFactory relBuilderFactory) {
		this(
				operandJ(filterClass, null, filterPredicate,
						operandJ(projectClass, null, projectPredicate, any())),
				copyFilter, copyProject, relBuilderFactory);
	}

	@Deprecated // to be removed before 2.0
	public FilterProjectTransposeRule(
			Class<? extends Filter> filterClass,
			RelFactories.FilterFactory filterFactory,
			Class<? extends Project> projectClass,
			RelFactories.ProjectFactory projectFactory) {
		this(filterClass, filter -> !RexUtil.containsCorrelation(filter.getCondition()),
				projectClass, project -> true,
				filterFactory == null,
				projectFactory == null,
				RelBuilder.proto(filterFactory, projectFactory));
	}

	protected FilterProjectTransposeRule(
			RelOptRuleOperand operand,
			boolean copyFilter,
			boolean copyProject,
			RelBuilderFactory relBuilderFactory) {
		super(operand, relBuilderFactory, null);
		this.copyFilter = copyFilter;
		this.copyProject = copyProject;
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public void onMatch(RelOptRuleCall call) {
		final Filter filter = call.rel(0);
		final Project project = call.rel(1);

		if (RexOver.containsOver(project.getProjects(), null)) {
			// In general a filter cannot be pushed below a windowing calculation.
			// Applying the filter before the aggregation function changes
			// the results of the windowing invocation.
			//
			// When the filter is on the PARTITION BY expression of the OVER clause
			// it can be pushed down. For now we don't support this.
			return;
		}
		// convert the filter to one that references the child of the project
		RexNode newCondition =
				RelOptUtil.pushPastProject(filter.getCondition(), project);

		final RelBuilder relBuilder = call.builder();
		RelNode newFilterRel;
		if (copyFilter) {
			final RelNode input = project.getInput();
			final RelTraitSet traitSet = filter.getTraitSet()
					.replaceIfs(RelCollationTraitDef.INSTANCE,
							() -> Collections.singletonList(
									input.getTraitSet().getTrait(RelCollationTraitDef.INSTANCE)))
					.replaceIfs(RelDistributionTraitDef.INSTANCE,
							() -> Collections.singletonList(
									input.getTraitSet().getTrait(RelDistributionTraitDef.INSTANCE)));
			newCondition = RexUtil.removeNullabilityCast(relBuilder.getTypeFactory(), newCondition);
			newFilterRel = filter.copy(traitSet, input, newCondition);
		} else {
			newFilterRel =
					relBuilder.push(project.getInput()).filter(newCondition).build();
		}

		RelNode newProjRel =
				copyProject
				? project.copy(project.getTraitSet(), newFilterRel,
						project.getProjects(), project.getRowType())
						: relBuilder.push(newFilterRel)
						.project(project.getProjects(), project.getRowType().getFieldNames())
						.build();
				call.transformTo(newProjRel);
	}
}
