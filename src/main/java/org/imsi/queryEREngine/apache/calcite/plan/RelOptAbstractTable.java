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
package org.imsi.queryEREngine.apache.calcite.plan;

import java.util.Collections;
import java.util.List;

import org.apache.calcite.linq4j.tree.Expression;
import org.imsi.queryEREngine.apache.calcite.prepare.RelOptTableImpl;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelDistribution;
import org.imsi.queryEREngine.apache.calcite.rel.RelDistributions;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelReferentialConstraint;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableScan;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.schema.ColumnStrategy;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

/**
 * Partial implementation of {@link RelOptTable}.
 */
public abstract class RelOptAbstractTable implements RelOptTable {
	//~ Instance fields --------------------------------------------------------

	protected final RelOptSchema schema;
	protected final RelDataType rowType;
	protected final String name;

	//~ Constructors -----------------------------------------------------------

	protected RelOptAbstractTable(
			RelOptSchema schema,
			String name,
			RelDataType rowType) {
		this.schema = schema;
		this.name = name;
		this.rowType = rowType;
	}

	//~ Methods ----------------------------------------------------------------

	public String getName() {
		return name;
	}

	@Override
	public List<String> getQualifiedName() {
		return ImmutableList.of(name);
	}

	@Override
	public double getRowCount() {
		return 100;
	}

	@Override
	public RelDataType getRowType() {
		return rowType;
	}

	@Override
	public RelOptSchema getRelOptSchema() {
		return schema;
	}

	// Override to define collations.
	@Override
	public List<RelCollation> getCollationList() {
		return Collections.emptyList();
	}

	@Override
	public RelDistribution getDistribution() {
		return RelDistributions.BROADCAST_DISTRIBUTED;
	}

	@Override
	public <T> T unwrap(Class<T> clazz) {
		return clazz.isInstance(this)
				? clazz.cast(this)
						: null;
	}

	// Override to define keys
	@Override
	public boolean isKey(ImmutableBitSet columns) {
		return false;
	}

	// Override to get unique keys
	@Override
	public List<ImmutableBitSet> getKeys() {
		return Collections.emptyList();
	}

	// Override to define foreign keys
	@Override
	public List<RelReferentialConstraint> getReferentialConstraints() {
		return Collections.emptyList();
	}

	@Override
	public RelNode toRel(ToRelContext context) {
		return LogicalTableScan.create(context.getCluster(), this,
				context.getTableHints());
	}

	@Override
	public Expression getExpression(Class clazz) {
		return null;
	}

	@Override
	public RelOptTable extend(List<RelDataTypeField> extendedFields) {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<ColumnStrategy> getColumnStrategies() {
		return RelOptTableImpl.columnStrategies(this);
	}

}
