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
package org.imsi.queryEREngine.apache.calcite.rel.logical;

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.plan.Convention;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollationTraitDef;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollations;
import org.imsi.queryEREngine.apache.calcite.rel.RelInput;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelShuttle;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.hint.RelHint;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMdCollation;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorUtil;
import org.imsi.queryEREngine.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;

/**
 * Sub-class of {@link org.imsi.queryEREngine.apache.calcite.rel.core.Project} not
 * targeted at any particular engine or calling convention.
 */
public final class LogicalProject extends Project {
	//~ Constructors -----------------------------------------------------------


	/**
	 * Creates a LogicalProject.
	 *
	 * <p>Use {@link #create} unless you know what you're doing.
	 *
	 * @param cluster  Cluster this relational expression belongs to
	 * @param traitSet Traits of this relational expression
	 * @param hints    Hints of this relational expression
	 * @param input    Input relational expression
	 * @param projects List of expressions for the input columns
	 * @param rowType  Output row type
	 */


	public LogicalProject(
			RelOptCluster cluster,
			RelTraitSet traitSet,
			List<RelHint> hints,
			RelNode input,
			List<? extends RexNode> projects,
			RelDataType rowType) {
		super(cluster, traitSet, hints, input, projects, rowType);
		assert traitSet.containsIfApplicable(Convention.NONE);
	}


	@Deprecated // to be removed before 2.0
	public LogicalProject(RelOptCluster cluster, RelTraitSet traitSet,
			RelNode input, List<? extends RexNode> projects, RelDataType rowType) {
		this(cluster, traitSet, ImmutableList.of(), input, projects, rowType);
	}

	@Deprecated // to be removed before 2.0
	public LogicalProject(RelOptCluster cluster, RelTraitSet traitSet,
			RelNode input, List<? extends RexNode> projects, RelDataType rowType,
			int flags) {
		this(cluster, traitSet, ImmutableList.of(), input, projects, rowType);
		Util.discard(flags);
	}

	@Deprecated // to be removed before 2.0
	public LogicalProject(RelOptCluster cluster, RelNode input,
			List<RexNode> projects, List<String> fieldNames, int flags) {
		this(cluster, cluster.traitSetOf(RelCollations.EMPTY),
				ImmutableList.of(), input, projects,
				RexUtil.createStructType(cluster.getTypeFactory(), projects,
						fieldNames, null));
		Util.discard(flags);
	}

	/**
	 * Creates a LogicalProject by parsing serialized output.
	 */
	public LogicalProject(RelInput input) {
		super(input);
	}

	//~ Methods ----------------------------------------------------------------



	/** Creates a LogicalProject. */
	public static LogicalProject create(final RelNode input, List<RelHint> hints,
			final List<? extends RexNode> projects, List<String> fieldNames) {
		final RelOptCluster cluster = input.getCluster();
		final RelDataType rowType =
				RexUtil.createStructType(cluster.getTypeFactory(), projects,
						fieldNames, SqlValidatorUtil.F_SUGGESTER);
		return create(input, hints, projects, rowType);
	}

	/** Creates a LogicalProject, specifying row type rather than field names. */
	public static LogicalProject create(final RelNode input, List<RelHint> hints,
			final List<? extends RexNode> projects, RelDataType rowType) {
		final RelOptCluster cluster = input.getCluster();
		final RelMetadataQuery mq = cluster.getMetadataQuery();
		final RelTraitSet traitSet =
				cluster.traitSet().replace(Convention.NONE)
				.replaceIfs(RelCollationTraitDef.INSTANCE,
						() -> RelMdCollation.project(mq, input, projects));
		return new LogicalProject(cluster, traitSet, hints, input, projects, rowType);
	}


	@Deprecated // to be removed before 1.23
	public static LogicalProject create(final RelNode input,
			final List<? extends RexNode> projects, List<String> fieldNames) {
		return create(input, ImmutableList.of(), projects, fieldNames);
	}

	@Deprecated // to be removed before 1.23
	public static LogicalProject create(final RelNode input,
			final List<? extends RexNode> projects, RelDataType rowType) {
		return create(input, ImmutableList.of(), projects, rowType);
	}

	@Override public LogicalProject copy(RelTraitSet traitSet, RelNode input,
			List<RexNode> projects, RelDataType rowType) {
		return new LogicalProject(getCluster(), traitSet, hints, input, projects, rowType);
	}

	@Override public RelNode accept(RelShuttle shuttle) {
		return shuttle.visit(this);
	}

	@Override public RelNode withHints(List<RelHint> hintList) {
		return new LogicalProject(getCluster(), traitSet, hintList,
				input, getProjects(), rowType);
	}


}
