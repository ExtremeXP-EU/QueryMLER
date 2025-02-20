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

import java.util.Set;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.CorrelationId;
import org.imsi.queryEREngine.apache.calcite.rel.core.JoinRelType;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableIntList;

/**
 * Base class for any join whose condition is based on column equality.
 *
 * @deprecated Use
 * {@link org.imsi.queryEREngine.apache.calcite.rel.core.EquiJoin EquiJoin in 'core' package}
 */
@Deprecated // to be removed before 2.0
public abstract class EquiJoin extends org.imsi.queryEREngine.apache.calcite.rel.core.EquiJoin {
	public EquiJoin(RelOptCluster cluster, RelTraitSet traits, RelNode left,
			RelNode right, RexNode condition, ImmutableIntList leftKeys,
			ImmutableIntList rightKeys, JoinRelType joinType,
			Set<String> variablesStopped) {
		super(cluster, traits, left, right, condition, leftKeys, rightKeys,
				CorrelationId.setOf(variablesStopped), joinType);
	}
}
