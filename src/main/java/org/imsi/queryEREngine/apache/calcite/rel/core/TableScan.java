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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCost;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.AbstractRelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelInput;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelShuttle;
import org.imsi.queryEREngine.apache.calcite.rel.RelWriter;
import org.imsi.queryEREngine.apache.calcite.rel.hint.Hintable;
import org.imsi.queryEREngine.apache.calcite.rel.hint.RelHint;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableBitSet;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableIntList;

import com.google.common.collect.ImmutableList;

/**
 * Relational operator that returns the contents of a table.
 */
public abstract class TableScan
extends AbstractRelNode implements Hintable {
	//~ Instance fields --------------------------------------------------------

	/**
	 * The table definition.
	 */
	protected final RelOptTable table;

	/**
	 * The table hints.
	 */
	protected final ImmutableList<RelHint> hints;

	//~ Constructors -----------------------------------------------------------

	protected TableScan(RelOptCluster cluster, RelTraitSet traitSet,
			List<RelHint> hints, RelOptTable table) {
		super(cluster, traitSet);
		this.table = table;
		if (table.getRelOptSchema() != null) {
			cluster.getPlanner().registerSchema(table.getRelOptSchema());
		}
		this.hints = ImmutableList.copyOf(hints);
	}

	@Deprecated // to be removed before 2.0
	protected TableScan(RelOptCluster cluster, RelTraitSet traitSet,
			RelOptTable table) {
		this(cluster, traitSet, ImmutableList.of(), table);
	}

	/**
	 * Creates a TableScan by parsing serialized output.
	 */
	protected TableScan(RelInput input) {
		this(input.getCluster(), input.getTraitSet(), ImmutableList.of(), input.getTable("table"));
	}

	//~ Methods ----------------------------------------------------------------

	@Override public double estimateRowCount(RelMetadataQuery mq) {
		return table.getRowCount();
	}

	@Override public RelOptTable getTable() {
		return table;
	}

	@SuppressWarnings("deprecation")
	@Override public List<RelCollation> getCollationList() {
		return table.getCollationList();
	}

	@Override public RelOptCost computeSelfCost(RelOptPlanner planner,
			RelMetadataQuery mq) {
		double dRows = table.getRowCount();
		double dCpu = dRows + 1; // ensure non-zero cost
		double dIo = 0;
		return planner.getCostFactory().makeCost(dRows, dCpu, dIo);
	}

	@Override public RelDataType deriveRowType() {
		return table.getRowType();
	}

	/** Returns an identity projection for the given table. */
	public static ImmutableIntList identity(RelOptTable table) {
		return ImmutableIntList.identity(table.getRowType().getFieldCount());
	}

	/** Returns an identity projection. */
	public ImmutableIntList identity() {
		return identity(table);
	}

	@Override public RelWriter explainTerms(RelWriter pw) {
		return super.explainTerms(pw)
				.item("table", table.getQualifiedName());
	}

	/**
	 * Projects a subset of the fields of the table, and also asks for "extra"
	 * fields that were not included in the table's official type.
	 *
	 * <p>The default implementation assumes that tables cannot do either of
	 * these operations, therefore it adds a {@link Project} that projects
	 * {@code NULL} values for the extra fields, using the
	 * {@link RelBuilder#project(Iterable)} method.
	 *
	 * <p>Sub-classes, representing table types that have these capabilities,
	 * should override.</p>
	 *
	 * @param fieldsUsed  Bitmap of the fields desired by the consumer
	 * @param extraFields Extra fields, not advertised in the table's row-type,
	 *                    wanted by the consumer
	 * @param relBuilder Builder used to create a Project
	 * @return Relational expression that projects the desired fields
	 */
	public RelNode project(ImmutableBitSet fieldsUsed,
			Set<RelDataTypeField> extraFields,
			RelBuilder relBuilder) {
		final int fieldCount = getRowType().getFieldCount();
		if (fieldsUsed.equals(ImmutableBitSet.range(fieldCount))
				&& extraFields.isEmpty()) {
			return this;
		}
		final List<RexNode> exprList = new ArrayList<>();
		final List<String> nameList = new ArrayList<>();
		final RexBuilder rexBuilder = getCluster().getRexBuilder();
		final List<RelDataTypeField> fields = getRowType().getFieldList();

		// Project the subset of fields.
		for (int i : fieldsUsed) {
			RelDataTypeField field = fields.get(i);
			exprList.add(rexBuilder.makeInputRef(this, i));
			nameList.add(field.getName());
		}

		// Project nulls for the extra fields. (Maybe a sub-class table has
		// extra fields, but we don't.)
		for (RelDataTypeField extraField : extraFields) {
			exprList.add(rexBuilder.makeNullLiteral(extraField.getType()));
			nameList.add(extraField.getName());
		}

		return relBuilder.push(this).project(exprList, nameList).build();
	}

	@Override public RelNode accept(RelShuttle shuttle) {
		return shuttle.visit(this);
	}

	@Override public ImmutableList<RelHint> getHints() {
		return hints;
	}
}
