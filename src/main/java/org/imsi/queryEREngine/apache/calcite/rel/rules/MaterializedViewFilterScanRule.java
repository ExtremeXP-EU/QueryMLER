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
import java.util.List;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptMaterialization;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptMaterializations;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRule;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptRuleCall;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.plan.SubstitutionVisitor;
import org.imsi.queryEREngine.apache.calcite.plan.hep.HepPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.hep.HepProgram;
import org.imsi.queryEREngine.apache.calcite.plan.hep.HepProgramBuilder;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableScan;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/**
 * Planner rule that converts
 * a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Filter}
 * on a {@link org.imsi.queryEREngine.apache.calcite.rel.core.TableScan}
 * to a {@link org.imsi.queryEREngine.apache.calcite.rel.core.Filter} on Materialized View
 */
public class MaterializedViewFilterScanRule extends RelOptRule {
	public static final MaterializedViewFilterScanRule INSTANCE =
			new MaterializedViewFilterScanRule(RelFactories.LOGICAL_BUILDER);

	private final HepProgram program = new HepProgramBuilder()
			.addRuleInstance(FilterProjectTransposeRule.INSTANCE)
			.addRuleInstance(ProjectMergeRule.INSTANCE)
			.build();

	//~ Constructors -----------------------------------------------------------

	/** Creates a MaterializedViewFilterScanRule. */
	public MaterializedViewFilterScanRule(RelBuilderFactory relBuilderFactory) {
		super(operand(Filter.class, operand(TableScan.class, null, none())),
				relBuilderFactory, "MaterializedViewFilterScanRule");
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public void onMatch(RelOptRuleCall call) {
		final Filter filter = call.rel(0);
		final TableScan scan = call.rel(1);
		apply(call, filter, scan);
	}

	protected void apply(RelOptRuleCall call, Filter filter, TableScan scan) {
		final RelOptPlanner planner = call.getPlanner();
		final List<RelOptMaterialization> materializations =
				planner.getMaterializations();
		if (!materializations.isEmpty()) {
			RelNode root = filter.copy(filter.getTraitSet(),
					Collections.singletonList((RelNode) scan));
			List<RelOptMaterialization> applicableMaterializations =
					RelOptMaterializations.getApplicableMaterializations(root, materializations);
			for (RelOptMaterialization materialization : applicableMaterializations) {
				if (RelOptUtil.areRowTypesEqual(scan.getRowType(),
						materialization.queryRel.getRowType(), false)) {
					RelNode target = materialization.queryRel;
					final HepPlanner hepPlanner =
							new HepPlanner(program, planner.getContext());
					hepPlanner.setRoot(target);
					target = hepPlanner.findBestExp();
					List<RelNode> subs = new SubstitutionVisitor(target, root)
							.go(materialization.tableRel);
					for (RelNode s : subs) {
						call.transformTo(s);
					}
				}
			}
		}
	}
}
