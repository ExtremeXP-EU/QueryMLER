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
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.convert.ConverterRule;
import org.imsi.queryEREngine.apache.calcite.rel.core.RelFactories;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalFilter;

/**
 * Rule to convert a {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalFilter} to an
 * {@link EnumerableFilter}.
 */
class EnumerableFilterRule extends ConverterRule {
	EnumerableFilterRule() {
		super(LogicalFilter.class,
				(Predicate<LogicalFilter>) RelOptUtil::containsMultisetOrWindowedAgg,
				Convention.NONE, EnumerableConvention.INSTANCE,
				RelFactories.LOGICAL_BUILDER, "EnumerableFilterRule");
	}

	@Override
	public RelNode convert(RelNode rel) {
		final LogicalFilter filter = (LogicalFilter) rel;
		return new EnumerableFilter(rel.getCluster(),
				rel.getTraitSet().replace(EnumerableConvention.INSTANCE),
				convert(filter.getInput(),
						filter.getInput().getTraitSet()
						.replace(EnumerableConvention.INSTANCE)),
				filter.getCondition());
	}
}
