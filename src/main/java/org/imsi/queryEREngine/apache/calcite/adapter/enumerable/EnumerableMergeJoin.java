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

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.imsi.queryEREngine.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.linq4j.tree.BlockBuilder;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.linq4j.tree.Expressions;
import org.apache.calcite.linq4j.tree.ParameterExpression;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCost;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollationTraitDef;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollations;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.CorrelationId;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.JoinRelType;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMdCollation;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.util.BuiltInMethod;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableIntList;
import org.imsi.queryEREngine.apache.calcite.util.Pair;
import org.imsi.queryEREngine.apache.calcite.util.Source;
import org.imsi.queryEREngine.imsi.calcite.adapter.enumerable.csv.CsvFieldType;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

/** Implementation of {@link org.imsi.queryEREngine.apache.calcite.rel.core.Join} in
 * {@link EnumerableConvention enumerable calling convention} using
 * a merge algorithm. */
public class EnumerableMergeJoin extends Join implements EnumerableRel {
	EnumerableMergeJoin(
			RelOptCluster cluster,
			RelTraitSet traits,
			RelNode left,
			RelNode right,
			RexNode condition,
			Set<CorrelationId> variablesSet,
			JoinRelType joinType) {
		super(cluster, traits, ImmutableList.of(), left, right, condition, variablesSet, joinType);
		final List<RelCollation> collations =
				traits.getTraits(RelCollationTraitDef.INSTANCE);
		assert collations == null || RelCollations.contains(collations, joinInfo.leftKeys);
	}

	@Deprecated // to be removed before 2.0
	EnumerableMergeJoin(RelOptCluster cluster, RelTraitSet traits, RelNode left,
			RelNode right, RexNode condition, ImmutableIntList leftKeys,
			ImmutableIntList rightKeys, Set<CorrelationId> variablesSet,
			JoinRelType joinType) {
		this(cluster, traits, left, right, condition, variablesSet, joinType);
	}

	@Deprecated // to be removed before 2.0
	EnumerableMergeJoin(RelOptCluster cluster, RelTraitSet traits, RelNode left,
			RelNode right, RexNode condition, ImmutableIntList leftKeys,
			ImmutableIntList rightKeys, JoinRelType joinType,
			Set<String> variablesStopped) {
		this(cluster, traits, left, right, condition, leftKeys, rightKeys,
				CorrelationId.setOf(variablesStopped), joinType);
	}

	public static EnumerableMergeJoin create(RelNode left, RelNode right,
			RexNode condition, ImmutableIntList leftKeys,
			ImmutableIntList rightKeys, JoinRelType joinType) {
		final RelOptCluster cluster = right.getCluster();
		RelTraitSet traitSet = cluster.traitSet();
		if (traitSet.isEnabled(RelCollationTraitDef.INSTANCE)) {
			final RelMetadataQuery mq = cluster.getMetadataQuery();
			final List<RelCollation> collations =
					RelMdCollation.mergeJoin(mq, left, right, leftKeys, rightKeys);
			traitSet = traitSet.replace(collations);
		}
		return new EnumerableMergeJoin(cluster, traitSet, left, right, condition,
				ImmutableSet.of(), joinType);
	}

	@Override public EnumerableMergeJoin copy(RelTraitSet traitSet,
			RexNode condition, RelNode left, RelNode right, JoinRelType joinType,
			boolean semiJoinDone) {
		return new EnumerableMergeJoin(getCluster(), traitSet, left, right,
				condition, variablesSet, joinType);
	}

	@Override public RelOptCost computeSelfCost(RelOptPlanner planner,
			RelMetadataQuery mq) {
		// We assume that the inputs are sorted. The price of sorting them has
		// already been paid. The cost of the join is therefore proportional to the
		// input and output size.
		final double rightRowCount = right.estimateRowCount(mq);
		final double leftRowCount = left.estimateRowCount(mq);
		final double rowCount = mq.getRowCount(this);
		final double d = leftRowCount + rightRowCount + rowCount;
		return planner.getCostFactory().makeCost(d, 0, 0);
	}

	@Override
	public Result implement(EnumerableRelImplementor implementor, Prefer pref) {
		BlockBuilder builder = new BlockBuilder();
		final Result leftResult =
				implementor.visitChild(this, 0, (EnumerableRel) left, pref);
		final Expression leftExpression =
				builder.append("left", leftResult.block);
		final ParameterExpression left_ =
				Expressions.parameter(leftResult.physType.getJavaRowType(), "left");
		final Result rightResult =
				implementor.visitChild(this, 1, (EnumerableRel) right, pref);
		final Expression rightExpression =
				builder.append("right", rightResult.block);
		final ParameterExpression right_ =
				Expressions.parameter(rightResult.physType.getJavaRowType(), "right");
		final JavaTypeFactory typeFactory = implementor.getTypeFactory();
		final PhysType physType =
				PhysTypeImpl.of(typeFactory, getRowType(), pref.preferArray());
		final List<Expression> leftExpressions = new ArrayList<>();
		final List<Expression> rightExpressions = new ArrayList<>();
		for (Pair<Integer, Integer> pair : Pair.zip(joinInfo.leftKeys, joinInfo.rightKeys)) {
			final RelDataType keyType =
					typeFactory.leastRestrictive(
							ImmutableList.of(
									left.getRowType().getFieldList().get(pair.left).getType(),
									right.getRowType().getFieldList().get(pair.right).getType()));
			final Type keyClass = typeFactory.getJavaClass(keyType);
			leftExpressions.add(
					EnumUtils.convert(
							leftResult.physType.fieldReference(left_, pair.left), keyClass));
			rightExpressions.add(
					EnumUtils.convert(
							rightResult.physType.fieldReference(right_, pair.right), keyClass));
		}
		final PhysType leftKeyPhysType =
				leftResult.physType.project(joinInfo.leftKeys, JavaRowFormat.LIST);
		final PhysType rightKeyPhysType =
				rightResult.physType.project(joinInfo.rightKeys, JavaRowFormat.LIST);
		return implementor.result(
				physType,
				builder.append(
						Expressions.call(
								BuiltInMethod.MERGE_JOIN.method,
								Expressions.list(
										leftExpression,
										rightExpression,
										Expressions.lambda(
												leftKeyPhysType.record(leftExpressions), left_),
										Expressions.lambda(
												rightKeyPhysType.record(rightExpressions), right_),
										EnumUtils.joinSelector(joinType,
												physType,
												ImmutableList.of(
														leftResult.physType, rightResult.physType)),
										Expressions.constant(
												joinType.generatesNullsOnLeft()),
										Expressions.constant(
												joinType.generatesNullsOnRight())))).toBlock());
	}


	@Override
	public Join copy(RelTraitSet traitSet, RexNode conditionExpr, RelNode left, RelNode right, JoinRelType joinType,
			Source sourceLeft, Source sourceRight, List<CsvFieldType> fieldTypesLeft,
			List<CsvFieldType> fieldTypesRight, boolean semiJoinDone, Integer keyLeft, Integer keyRight,
			String tableNameLeft, String tableNameRight, Integer fieldLeft, Integer fieldRight, Boolean isDirtyJoin) {
		// TODO Auto-generated method stub
		return null;
	}
}
