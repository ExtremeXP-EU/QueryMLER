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
package org.imsi.queryEREngine.apache.calcite.adapter.enumerable;

import java.util.function.Predicate;

import org.imsi.queryEREngine.apache.calcite.plan.Convention;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.convert.ConverterRule;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableFunctionScan;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilderFactory;

/** Planner rule that converts a
 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableFunctionScan}
 * relational expression
 * {@link org.imsi.queryEREngine.apache.calcite.adapter.enumerable.EnumerableConvention enumerable calling convention}. */
public class EnumerableTableFunctionScanRule extends ConverterRule {
	@Deprecated // to be removed before 2.0
	public EnumerableTableFunctionScanRule() {
		this(RelFactories.LOGICAL_BUILDER);
	}

	/**
	 * Creates an EnumerableTableFunctionScanRule.
	 *
	 * @param relBuilderFactory Builder for relational expressions
	 */
	public EnumerableTableFunctionScanRule(RelBuilderFactory relBuilderFactory) {
		super(LogicalTableFunctionScan.class, (Predicate<RelNode>) r -> true,
				Convention.NONE, EnumerableConvention.INSTANCE, relBuilderFactory,
				"EnumerableTableFunctionScanRule");
	}

	@Override public RelNode convert(RelNode rel) {
		final RelTraitSet traitSet =
				rel.getTraitSet().replace(EnumerableConvention.INSTANCE);
		LogicalTableFunctionScan tbl = (LogicalTableFunctionScan) rel;
		return new EnumerableTableFunctionScan(rel.getCluster(), traitSet,
				convertList(tbl.getInputs(), traitSet.getTrait(0)), tbl.getElementType(), tbl.getRowType(),
				tbl.getCall(), tbl.getColumnMappings());
	}
}
