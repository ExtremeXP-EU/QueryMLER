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
package org.imsi.queryEREngine.apache.calcite.rel.core;

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelInput;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMdUtil;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;

/**
 * Relational expression that returns the rows of its first input minus any
 * matching rows from its other inputs.
 *
 * <p>Corresponds to the SQL {@code EXCEPT} operator.
 *
 * <p>If "all" is true, then multiset subtraction is
 * performed; otherwise, set subtraction is performed (implying no duplicates in
 * the results).
 */
public abstract class Minus extends SetOp {
	public Minus(RelOptCluster cluster, RelTraitSet traits, List<RelNode> inputs,
			boolean all) {
		super(cluster, traits, inputs, SqlKind.EXCEPT, all);
	}

	/**
	 * Creates a Minus by parsing serialized output.
	 */
	protected Minus(RelInput input) {
		super(input);
	}

	@Override public double estimateRowCount(RelMetadataQuery mq) {
		return RelMdUtil.getMinusRowCount(mq, this);
	}
}
