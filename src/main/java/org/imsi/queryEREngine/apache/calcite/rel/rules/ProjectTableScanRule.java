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
import java.util.stream.Collectors;

import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.EnumerableInterpreter;
import org.imsi.queryEREngine.apache.calcite.interpreter.Bindables;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleOperand;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableScan;
import org.imsi.queryEREngine.apache.calcite.rex.RexInputRef;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.rex.RexVisitorImpl;
import org.imsi.queryEREngine.apache.calcite.schema.ProjectableFilterableTable;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;
import org.imsi.queryEREngine.apache.calcite.util.mapping.Mapping;
import org.imsi.queryEREngine.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableList;

/**
 * Planner rule that converts a {@link Project}
 * on a {@link org.imsi.queryEREngine.apache.calcite.rel.core.TableScan}
 * of a {@link org.imsi.queryEREngine.apache.calcite.schema.ProjectableFilterableTable}
 * to a {@link org.imsi.queryEREngine.apache.calcite.interpreter.Bindables.BindableTableScan}.
 *
 * <p>The {@link #INTERPRETER} variant allows an intervening
 * {@link org.imsi.queryEREngine.apache.calcite.adapter.enumerable.EnumerableInterpreter}.
 *
 * @see FilterTableScanRule
 */
public abstract class ProjectTableScanRule extends RelOptRule {
	@SuppressWarnings("Guava")
	@Deprecated // to be removed before 2.0
	public static final com.google.common.base.Predicate<TableScan> PREDICATE =
	ProjectTableScanRule::test;

	/** Rule that matches Project on TableScan. */
	public static final ProjectTableScanRule INSTANCE =
			new ProjectTableScanRule(
					operand(Project.class,
							operandJ(TableScan.class, null, ProjectTableScanRule::test,
									none())),
					RelFactories.LOGICAL_BUILDER,
					"ProjectScanRule") {
		@Override public void onMatch(RelOptRuleCall call) {
			final Project project = call.rel(0);
			final TableScan scan = call.rel(1);
			apply(call, project, scan);
		}
	};

	/** Rule that matches Project on EnumerableInterpreter on TableScan. */
	public static final ProjectTableScanRule INTERPRETER =
			new ProjectTableScanRule(
					operand(Project.class,
							operand(EnumerableInterpreter.class,
									operandJ(TableScan.class, null, ProjectTableScanRule::test,
											none()))),
					RelFactories.LOGICAL_BUILDER,
					"ProjectScanRule:interpreter") {
		@Override public void onMatch(RelOptRuleCall call) {
			final Project project = call.rel(0);
			final TableScan scan = call.rel(2);
			apply(call, project, scan);
		}
	};

	//~ Constructors -----------------------------------------------------------

	/** Creates a ProjectTableScanRule. */
	public ProjectTableScanRule(RelOptRuleOperand operand,
			RelBuilderFactory relBuilderFactory, String description) {
		super(operand, relBuilderFactory, description);
	}

	//~ Methods ----------------------------------------------------------------

	protected static boolean test(TableScan scan) {
		// We can only push projects into a ProjectableFilterableTable.
		final RelOptTable table = scan.getTable();
		return table.unwrap(ProjectableFilterableTable.class) != null;
	}

	protected void apply(RelOptRuleCall call, Project project, TableScan scan) {
		final RelOptTable table = scan.getTable();
		assert table.unwrap(ProjectableFilterableTable.class) != null;

		final List<Integer> selectedColumns = new ArrayList<>();
		project.getProjects().forEach(proj -> {
			proj.accept(new RexVisitorImpl<Void>(true) {
				@Override
				public Void visitInputRef(RexInputRef inputRef) {
					if (!selectedColumns.contains(inputRef.getIndex())) {
						selectedColumns.add(inputRef.getIndex());
					}
					return null;
				}
			});
		});

		final List<RexNode> filtersPushDown;
		final List<Integer> projectsPushDown;
		if (scan instanceof Bindables.BindableTableScan) {
			final Bindables.BindableTableScan bindableScan =
					(Bindables.BindableTableScan) scan;
			filtersPushDown = bindableScan.filters;
			projectsPushDown = selectedColumns.stream()
					.map(col -> bindableScan.projects.get(col))
					.collect(Collectors.toList());
		} else {
			filtersPushDown = ImmutableList.of();
			projectsPushDown = selectedColumns;
		}
		Bindables.BindableTableScan newScan = Bindables.BindableTableScan.create(
				scan.getCluster(), scan.getTable(), filtersPushDown, projectsPushDown);
		Mapping mapping =
				Mappings.target(selectedColumns, scan.getRowType().getFieldCount());
		final List<RexNode> newProjectRexNodes =
				ImmutableList.copyOf(RexUtil.apply(mapping, project.getProjects()));

		if (RexUtil.isIdentity(newProjectRexNodes, newScan.getRowType())) {
			call.transformTo(newScan);
		} else {
			call.transformTo(
					call.builder()
					.push(newScan)
					.project(newProjectRexNodes)
					.build());
		}
	}
}
