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

import java.util.Objects;

import org.apache.calcite.linq4j.function.Experimental;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelWriter;

/**
 * Spool that writes into a table.
 *
 * <p>NOTE: The current API is experimental and subject to change without
 * notice.
 */
@Experimental
public abstract class TableSpool extends Spool {

	protected final RelOptTable table;

	protected TableSpool(RelOptCluster cluster, RelTraitSet traitSet,
			RelNode input, Type readType, Type writeType, RelOptTable table) {
		super(cluster, traitSet, input, readType, writeType);
		this.table = Objects.requireNonNull(table);
	}

	@Override
	public RelOptTable getTable() {
		return table;
	}

	@Override public RelWriter explainTerms(RelWriter pw) {
		super.explainTerms(pw);
		return pw.item("table", table.getQualifiedName());
	}
}
