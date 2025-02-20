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

import org.imsi.queryEREngine.apache.calcite.plan.Convention;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.InvalidRelException;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.convert.ConverterRule;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalAggregate;

/**
 * Rule to convert a {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalAggregate}
 * to an {@link EnumerableAggregate}.
 */
class EnumerableAggregateRule extends ConverterRule {
	EnumerableAggregateRule() {
		super(LogicalAggregate.class, Convention.NONE,
				EnumerableConvention.INSTANCE, "EnumerableAggregateRule");
	}

	@Override
	public RelNode convert(RelNode rel) {
		final LogicalAggregate agg = (LogicalAggregate) rel;
		final RelTraitSet traitSet =
				agg.getTraitSet().replace(EnumerableConvention.INSTANCE);
		try {
			return new EnumerableAggregate(
					rel.getCluster(),
					traitSet,
					convert(agg.getInput(), EnumerableConvention.INSTANCE),
					agg.getGroupSet(),
					agg.getGroupSets(),
					agg.getAggCallList());
		} catch (InvalidRelException e) {
			EnumerableRules.LOGGER.debug(e.toString());
			return null;
		}
	}
}
