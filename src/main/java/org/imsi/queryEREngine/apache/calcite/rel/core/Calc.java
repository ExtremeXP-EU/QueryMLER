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
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCost;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelWriter;
import org.imsi.queryEREngine.apache.calcite.rel.SingleRel;
import org.imsi.queryEREngine.apache.calcite.rel.hint.Hintable;
import org.imsi.queryEREngine.apache.calcite.rel.hint.RelHint;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMdUtil;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexLocalRef;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexProgram;
import org.imsi.queryEREngine.apache.calcite.rex.RexProgramBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexShuttle;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.util.Litmus;
import org.imsi.queryEREngine.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;

/**
 * <code>Calc</code> is an abstract base class for implementations of
 * {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalCalc}.
 */
public abstract class Calc extends SingleRel implements Hintable {
	//~ Instance fields --------------------------------------------------------

	protected final ImmutableList<RelHint> hints;

	protected final RexProgram program;

	//~ Constructors -----------------------------------------------------------
	/**
	 * Creates a Calc.
	 *
	 * @param cluster Cluster
	 * @param traits Traits
	 * @param hints Hints of this relational expression
	 * @param child Input relation
	 * @param program Calc program
	 */
	protected Calc(
			RelOptCluster cluster,
			RelTraitSet traits,
			List<RelHint> hints,
			RelNode child,
			RexProgram program) {
		super(cluster, traits, child);
		this.rowType = program.getOutputRowType();
		this.program = program;
		this.hints = ImmutableList.copyOf(hints);
		assert isValid(Litmus.THROW, null);
	}

	@Deprecated // to be removed before 2.0
	protected Calc(
			RelOptCluster cluster,
			RelTraitSet traits,
			RelNode child,
			RexProgram program) {
		this(cluster, traits, ImmutableList.of(), child, program);
	}

	@Deprecated // to be removed before 2.0
	protected Calc(
			RelOptCluster cluster,
			RelTraitSet traits,
			RelNode child,
			RexProgram program,
			List<RelCollation> collationList) {
		this(cluster, traits, ImmutableList.of(), child, program);
		Util.discard(collationList);
	}

	//~ Methods ----------------------------------------------------------------

	@Override public final Calc copy(RelTraitSet traitSet, List<RelNode> inputs) {
		return copy(traitSet, sole(inputs), program);
	}

	/**
	 * Creates a copy of this {@code Calc}.
	 *
	 * @param traitSet Traits
	 * @param child Input relation
	 * @param program Calc program
	 * @return New {@code Calc} if any parameter differs from the value of this
	 *   {@code Calc}, or just {@code this} if all the parameters are the same
	 *
	 * @see #copy(org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet, java.util.List)
	 */
	public abstract Calc copy(
			RelTraitSet traitSet,
			RelNode child,
			RexProgram program);

	@Deprecated // to be removed before 2.0
	public Calc copy(
			RelTraitSet traitSet,
			RelNode child,
			RexProgram program,
			List<RelCollation> collationList) {
		Util.discard(collationList);
		return copy(traitSet, child, program);
	}

	@Override
	public boolean isValid(Litmus litmus, Context context) {
		if (!RelOptUtil.equal(
				"program's input type",
				program.getInputRowType(),
				"child's output type",
				getInput().getRowType(), litmus)) {
			return litmus.fail(null);
		}
		if (!program.isValid(litmus, context)) {
			return litmus.fail(null);
		}
		if (!program.isNormalized(litmus, getCluster().getRexBuilder())) {
			return litmus.fail(null);
		}
		return litmus.succeed();
	}

	public RexProgram getProgram() {
		return program;
	}

	@Override public ImmutableList<RelHint> getHints() {
		return hints;
	}

	@Override public double estimateRowCount(RelMetadataQuery mq) {
		return RelMdUtil.estimateFilteredRows(getInput(), program, mq);
	}

	@Override public RelOptCost computeSelfCost(RelOptPlanner planner,
			RelMetadataQuery mq) {
		double dRows = mq.getRowCount(this);
		double dCpu = mq.getRowCount(getInput())
				* program.getExprCount();
		double dIo = 0;
		return planner.getCostFactory().makeCost(dRows, dCpu, dIo);
	}

	@Override
	public RelWriter explainTerms(RelWriter pw) {
		return program.explainCalc(super.explainTerms(pw));
	}

	@Override
	public RelNode accept(RexShuttle shuttle) {
		List<RexNode> oldExprs = program.getExprList();
		List<RexNode> exprs = shuttle.apply(oldExprs);
		List<RexLocalRef> oldProjects = program.getProjectList();
		List<RexLocalRef> projects = shuttle.apply(oldProjects);
		RexLocalRef oldCondition = program.getCondition();
		RexNode condition;
		if (oldCondition != null) {
			condition = shuttle.apply(oldCondition);
			assert condition instanceof RexLocalRef
			: "Invalid condition after rewrite. Expected RexLocalRef, got "
			+ condition;
		} else {
			condition = null;
		}
		if (exprs == oldExprs
				&& projects == oldProjects
				&& condition == oldCondition) {
			return this;
		}

		final RexBuilder rexBuilder = getCluster().getRexBuilder();
		final RelDataType rowType =
				RexUtil.createStructType(
						rexBuilder.getTypeFactory(),
						projects,
						this.rowType.getFieldNames(),
						null);
		final RexProgram newProgram =
				RexProgramBuilder.create(
						rexBuilder, program.getInputRowType(), exprs, projects,
						condition, rowType, true, null)
				.getProgram(false);
		return copy(traitSet, getInput(), newProgram);
	}
}
