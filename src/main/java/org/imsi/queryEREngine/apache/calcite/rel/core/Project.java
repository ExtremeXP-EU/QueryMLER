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

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.calcite.linq4j.Ord;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCost;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelInput;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelWriter;
import org.imsi.queryEREngine.apache.calcite.rel.SingleRel;
import org.imsi.queryEREngine.apache.calcite.rel.hint.Hintable;
import org.imsi.queryEREngine.apache.calcite.rel.hint.RelHint;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.rex.RexChecker;
import org.imsi.queryEREngine.apache.calcite.rex.RexInputRef;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexShuttle;
import org.imsi.queryEREngine.apache.calcite.rex.RexUtil;
import org.imsi.queryEREngine.apache.calcite.sql.SqlExplainLevel;
import org.imsi.queryEREngine.apache.calcite.util.Litmus;
import org.imsi.queryEREngine.apache.calcite.util.Pair;
import org.imsi.queryEREngine.apache.calcite.util.Permutation;
import org.imsi.queryEREngine.apache.calcite.util.Util;
import org.imsi.queryEREngine.apache.calcite.util.mapping.MappingType;
import org.imsi.queryEREngine.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;

/**
 * Relational expression that computes a set of
 * 'select expressions' from its input relational expression.
 *
 * @see org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject
 */
public abstract class Project extends SingleRel implements Hintable {
	//~ Instance fields --------------------------------------------------------

	protected final ImmutableList<RexNode> exps;

	protected final ImmutableList<RelHint> hints;

	//~ Constructors -----------------------------------------------------------

	/**
	 * Creates a Project.
	 *
	 * @param cluster  Cluster that this relational expression belongs to
	 * @param traits   Traits of this relational expression
	 * @param hints    Hints of this relation expression
	 * @param input    Input relational expression
	 * @param projects List of expressions for the input columns
	 * @param rowType  Output row type
	 */
	protected Project(
			RelOptCluster cluster,
			RelTraitSet traits,
			List<RelHint> hints,
			RelNode input,
			List<? extends RexNode> projects,
			RelDataType rowType) {
		super(cluster, traits, input);
		assert rowType != null;
		this.exps = ImmutableList.copyOf(projects);
		this.hints = ImmutableList.copyOf(hints);
		this.rowType = rowType;
		assert isValid(Litmus.THROW, null);
	}

	@Deprecated // to be removed before 2.0
	protected Project(RelOptCluster cluster, RelTraitSet traits,
			RelNode input, List<? extends RexNode> projects, RelDataType rowType) {
		this(cluster, traits, ImmutableList.of(), input, projects, rowType);
	}

	@Deprecated // to be removed before 2.0
	protected Project(RelOptCluster cluster, RelTraitSet traitSet, RelNode input,
			List<? extends RexNode> projects, RelDataType rowType, int flags) {
		this(cluster, traitSet, ImmutableList.of(), input, projects, rowType);
		Util.discard(flags);
	}

	/**
	 * Creates a Project by parsing serialized output.
	 */
	protected Project(RelInput input) {
		this(input.getCluster(),
				input.getTraitSet(),
				ImmutableList.of(),
				input.getInput(),
				input.getExpressionList("exprs"),
				input.getRowType("exprs", "fields"));
	}

	//~ Methods ----------------------------------------------------------------

	@Override public final RelNode copy(RelTraitSet traitSet,
			List<RelNode> inputs) {
		return copy(traitSet, sole(inputs), exps, rowType);
	}

	/**
	 * Copies a project.
	 *
	 * @param traitSet Traits
	 * @param input Input
	 * @param projects Project expressions
	 * @param rowType Output row type
	 * @return New {@code Project} if any parameter differs from the value of this
	 *   {@code Project}, or just {@code this} if all the parameters are
	 *   the same
	 *
	 * @see #copy(RelTraitSet, List)
	 */
	public abstract Project copy(RelTraitSet traitSet, RelNode input,
			List<RexNode> projects, RelDataType rowType);

	@Deprecated // to be removed before 2.0
	public Project copy(RelTraitSet traitSet, RelNode input,
			List<RexNode> projects, RelDataType rowType, int flags) {
		Util.discard(flags);
		return copy(traitSet, input, projects, rowType);
	}

	@Deprecated // to be removed before 2.0
	public boolean isBoxed() {
		return true;
	}

	@Override public List<RexNode> getChildExps() {
		return exps;
	}

	@Override
	public RelNode accept(RexShuttle shuttle) {
		List<RexNode> exps = shuttle.apply(this.exps);
		if (this.exps == exps) {
			return this;
		}
		final RelDataType rowType =
				RexUtil.createStructType(
						getInput().getCluster().getTypeFactory(),
						exps,
						this.rowType.getFieldNames(),
						null);
		return copy(traitSet, getInput(), exps, rowType);
	}

	/**
	 * Returns the project expressions.
	 *
	 * @return Project expressions
	 */
	public List<RexNode> getProjects() {
		return exps;
	}

	/**
	 * Returns a list of (expression, name) pairs. Convenient for various
	 * transformations.
	 *
	 * @return List of (expression, name) pairs
	 */
	public final List<Pair<RexNode, String>> getNamedProjects() {
		return Pair.zip(getProjects(), getRowType().getFieldNames());
	}

	@Override public ImmutableList<RelHint> getHints() {
		return hints;
	}

	@Deprecated // to be removed before 2.0
	public int getFlags() {
		return 1;
	}

	@Override
	public boolean isValid(Litmus litmus, Context context) {
		if (!super.isValid(litmus, context)) {
			return litmus.fail(null);
		}
		if (!RexUtil.compatibleTypes(exps, getRowType(), litmus)) {
			return litmus.fail("incompatible types");
		}
		RexChecker checker =
				new RexChecker(
						getInput().getRowType(), context, litmus);
		for (RexNode exp : exps) {
			exp.accept(checker);
			if (checker.getFailureCount() > 0) {
				return litmus.fail("{} failures in expression {}",
						checker.getFailureCount(), exp);
			}
		}
		if (!Util.isDistinct(rowType.getFieldNames())) {
			return litmus.fail("field names not distinct: {}", rowType);
		}
		//CHECKSTYLE: IGNORE 1
		if (false && !Util.isDistinct(Lists.transform(exps, RexNode::toString))) {
			// Projecting the same expression twice is usually a bad idea,
			// because it may create expressions downstream which are equivalent
			// but which look different. We can't ban duplicate projects,
			// because we need to allow
			//
			//  SELECT a, b FROM c UNION SELECT x, x FROM z
			return litmus.fail("duplicate expressions: {}", exps);
		}
		return litmus.succeed();
	}

	@Override public RelOptCost computeSelfCost(RelOptPlanner planner,
			RelMetadataQuery mq) {
		double dRows = mq.getRowCount(getInput());
		double dCpu = dRows * exps.size();
		double dIo = 0;
		return planner.getCostFactory().makeCost(dRows, dCpu, dIo);
	}

	/**
	 * Returns the number of expressions at the front of an array which are
	 * simply projections of the same field.
	 *
	 * @param refs References
	 * @return the index of the first non-trivial expression, or list.size otherwise
	 */
	private static int countTrivial(List<RexNode> refs) {
		for (int i = 0; i < refs.size(); i++) {
			RexNode ref = refs.get(i);
			if (!(ref instanceof RexInputRef)
					|| ((RexInputRef) ref).getIndex() != i) {
				return i;
			}
		}
		return refs.size();
	}

	@Override
	public RelWriter explainTerms(RelWriter pw) {
		super.explainTerms(pw);
		// Skip writing field names so the optimizer can reuse the projects that differ in
		// field names only
		if (pw.getDetailLevel() == SqlExplainLevel.DIGEST_ATTRIBUTES) {
			final int firstNonTrivial = countTrivial(exps);
			if (firstNonTrivial == 1) {
				pw.item("inputs", "0");
			} else if (firstNonTrivial != 0) {
				pw.item("inputs", "0.." + (firstNonTrivial - 1));
			}
			if (firstNonTrivial != exps.size()) {
				pw.item("exprs", exps.subList(firstNonTrivial, exps.size()));
			}
			return pw;
		}

		if (pw.nest()) {
			pw.item("fields", rowType.getFieldNames());
			pw.item("exprs", exps);
		} else {
			for (Ord<RelDataTypeField> field : Ord.zip(rowType.getFieldList())) {
				String fieldName = field.e.getName();
				if (fieldName == null) {
					fieldName = "field#" + field.i;
				}
				pw.item(fieldName, exps.get(field.i));
			}
		}

		return pw;
	}

	/**
	 * Returns a mapping, or null if this projection is not a mapping.
	 *
	 * @return Mapping, or null if this projection is not a mapping
	 */
	public Mappings.TargetMapping getMapping() {
		return getMapping(getInput().getRowType().getFieldCount(), exps);
	}

	/**
	 * Returns a mapping of a set of project expressions.
	 *
	 * <p>The mapping is an inverse surjection.
	 * Every target has a source field, but no
	 * source has more than one target.
	 * Thus you can safely call
	 * {@link org.imsi.queryEREngine.apache.calcite.util.mapping.Mappings.TargetMapping#getSourceOpt(int)}.
	 *
	 * @param inputFieldCount Number of input fields
	 * @param projects Project expressions
	 * @return Mapping of a set of project expressions, or null if projection is
	 * not a mapping
	 */
	public static Mappings.TargetMapping getMapping(int inputFieldCount,
			List<? extends RexNode> projects) {
		if (inputFieldCount < projects.size()) {
			return null; // surjection is not possible
		}
		Mappings.TargetMapping mapping =
				Mappings.create(MappingType.INVERSE_SURJECTION,
						inputFieldCount, projects.size());
		for (Ord<RexNode> exp : Ord.<RexNode>zip(projects)) {
			if (!(exp.e instanceof RexInputRef)) {
				return null;
			}

			int source = ((RexInputRef) exp.e).getIndex();
			if (mapping.getTargetOpt(source) != -1) {
				return null;
			}
			mapping.set(source, exp.i);
		}
		return mapping;
	}

	/**
	 * Returns a partial mapping of a set of project expressions.
	 *
	 * <p>The mapping is an inverse function.
	 * Every target has a source field, but
	 * a source might have 0, 1 or more targets.
	 * Project expressions that do not consist of
	 * a mapping are ignored.
	 *
	 * @param inputFieldCount Number of input fields
	 * @param projects Project expressions
	 * @return Mapping of a set of project expressions, never null
	 */
	public static Mappings.TargetMapping getPartialMapping(int inputFieldCount,
			List<? extends RexNode> projects) {
		Mappings.TargetMapping mapping =
				Mappings.create(MappingType.INVERSE_FUNCTION,
						inputFieldCount, projects.size());
		for (Ord<RexNode> exp : Ord.<RexNode>zip(projects)) {
			if (exp.e instanceof RexInputRef) {
				mapping.set(((RexInputRef) exp.e).getIndex(), exp.i);
			}
		}
		return mapping;
	}

	/**
	 * Returns a permutation, if this projection is merely a permutation of its
	 * input fields; otherwise null.
	 *
	 * @return Permutation, if this projection is merely a permutation of its
	 *   input fields; otherwise null
	 */
	public Permutation getPermutation() {
		return getPermutation(getInput().getRowType().getFieldCount(), exps);
	}

	/**
	 * Returns a permutation, if this projection is merely a permutation of its
	 * input fields; otherwise null.
	 */
	public static Permutation getPermutation(int inputFieldCount,
			List<? extends RexNode> projects) {
		final int fieldCount = projects.size();
		if (fieldCount != inputFieldCount) {
			return null;
		}
		final Permutation permutation = new Permutation(fieldCount);
		final Set<Integer> alreadyProjected = new HashSet<>(fieldCount);
		for (int i = 0; i < fieldCount; ++i) {
			final RexNode exp = projects.get(i);
			if (exp instanceof RexInputRef) {
				final int index = ((RexInputRef) exp).getIndex();
				if (!alreadyProjected.add(index)) {
					return null;
				}
				permutation.set(i, index);
			} else {
				return null;
			}
		}
		return permutation;
	}

	/**
	 * Checks whether this is a functional mapping.
	 * Every output is a source field, but
	 * a source field may appear as zero, one, or more output fields.
	 */
	public boolean isMapping() {
		for (RexNode exp : exps) {
			if (!(exp instanceof RexInputRef)) {
				return false;
			}
		}
		return true;
	}

	//~ Inner Classes ----------------------------------------------------------

	/** No longer used. */
	@Deprecated // to be removed before 2.0
	public static class Flags {
		public static final int ANON_FIELDS = 2;
		public static final int BOXED = 1;
		public static final int NONE = 0;
	}
}
