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
import java.util.Objects;

import org.imsi.queryEREngine.apache.calcite.config.CalciteSystemProperty;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelWriter;
import org.imsi.queryEREngine.apache.calcite.rel.SingleRel;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexShuttle;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.util.Litmus;

import com.google.common.collect.ImmutableList;

/**
 * Relational expression that returns the contents of a relation expression as
 * it was at a given time in the past.
 *
 * <p>For example, if {@code Products} is a temporal table, and
 * {@link TableScan}(Products) is a relational operator that returns all
 * versions of the contents of the table, then
 * {@link Snapshot}(TableScan(Products)) is a relational operator that only
 * returns the contents whose versions that overlap with the given specific
 * period (i.e. those that started before given period and ended after it).
 */
public abstract class Snapshot extends SingleRel  {
	//~ Instance fields --------------------------------------------------------

	private final RexNode period;

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a Snapshot.
	 *
	 * @param cluster   Cluster that this relational expression belongs to
	 * @param traitSet  The traits of this relational expression
	 * @param input     Input relational expression
	 * @param period    Timestamp expression which as the table was at the given
	 *                  time in the past
	 */
	protected Snapshot(RelOptCluster cluster, RelTraitSet traitSet, RelNode input,
			RexNode period) {
		super(cluster, traitSet, input);
		this.period = Objects.requireNonNull(period);
		// Too expensive for everyday use:
		assert !CalciteSystemProperty.DEBUG.value() || isValid(Litmus.THROW, null);
	}

	//~ Methods ----------------------------------------------------------------

	@Override public final RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
		return copy(traitSet, sole(inputs), getPeriod());
	}

	public abstract Snapshot copy(RelTraitSet traitSet, RelNode input, RexNode period);

	@Override public List<RexNode> getChildExps() {
		return ImmutableList.of(period);
	}

	@Override
	public RelNode accept(RexShuttle shuttle) {
		RexNode condition = shuttle.apply(this.period);
		if (this.period == condition) {
			return this;
		}
		return copy(traitSet, getInput(), condition);
	}

	@Override public RelWriter explainTerms(RelWriter pw) {
		return super.explainTerms(pw)
				.item("period", period);
	}

	public RexNode getPeriod() {
		return period;
	}

	@Override public boolean isValid(Litmus litmus, Context context) {
		RelDataType dataType = period.getType();
		if (dataType.getSqlTypeName() != SqlTypeName.TIMESTAMP) {
			return litmus.fail("The system time period specification expects Timestamp type but is '"
					+ dataType.getSqlTypeName() + "'");
		}
		return litmus.succeed();
	}
}
