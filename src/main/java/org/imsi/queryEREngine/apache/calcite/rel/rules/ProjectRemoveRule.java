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

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/**
 * Planner rule that,
 * given a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Project} node that
 * merely returns its input, converts the node into its child.
 *
 * <p>For example, <code>Project(ArrayReader(a), {$input0})</code> becomes
 * <code>ArrayReader(a)</code>.</p>
 *
 * @see CalcRemoveRule
 * @see ProjectMergeRule
 */
public class ProjectRemoveRule extends RelOptRule {
	public static final ProjectRemoveRule INSTANCE =
			new ProjectRemoveRule(RelFactories.LOGICAL_BUILDER);

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a ProjectRemoveRule.
	 *
	 * @param relBuilderFactory Builder for relational expressions
	 */
	public ProjectRemoveRule(RelBuilderFactory relBuilderFactory) {
		// Create a specialized operand to detect non-matches early. This keeps
		// the rule queue short.
		super(operandJ(Project.class, null, ProjectRemoveRule::isTrivial, any()),
				relBuilderFactory, null);
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public void onMatch(RelOptRuleCall call) {
		Project project = call.rel(0);
		assert isTrivial(project);
		RelNode stripped = project.getInput();
		if (stripped instanceof Project) {
			// Rename columns of child projection if desired field names are given.
			Project childProject = (Project) stripped;
			stripped = childProject.copy(childProject.getTraitSet(),
					childProject.getInput(), childProject.getProjects(),
					project.getRowType());
		}
		RelNode child = call.getPlanner().register(stripped, project);
		call.transformTo(child);
	}

	/**
	 * Returns the child of a project if the project is trivial, otherwise
	 * the project itself.
	 */
	public static RelNode strip(Project project) {
		return isTrivial(project) ? project.getInput() : project;
	}

	public static boolean isTrivial(Project project) {
		return RexUtil.isIdentity(project.getProjects(),
				project.getInput().getRowType());
	}

	@Deprecated // to be removed before 1.5
	public static boolean isIdentity(List<? extends RexNode> exps,
			RelDataType childRowType) {
		return RexUtil.isIdentity(exps, childRowType);
	}
}
