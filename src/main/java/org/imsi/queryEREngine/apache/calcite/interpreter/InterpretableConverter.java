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
package org.imsi.queryEREngine.apache.calcite.interpreter;

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.DataContext;
import org.apache.calcite.linq4j.Enumerable;
import org.imsi.queryEREngine.apache.calcite.plan.ConventionTraitDef;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.convert.ConverterImpl;
import org.imsi.queryEREngine.apache.calcite.runtime.ArrayBindable;

/**
 * Relational expression that converts any relational expression input to
 * {@link org.imsi.queryEREngine.apache.calcite.interpreter.InterpretableConvention}, by wrapping
 * it in an interpreter.
 */
public class InterpretableConverter extends ConverterImpl
implements ArrayBindable {
	protected InterpretableConverter(RelOptCluster cluster, RelTraitSet traits,
			RelNode input) {
		super(cluster, ConventionTraitDef.INSTANCE, traits, input);
	}

	@Override public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
		return new InterpretableConverter(getCluster(), traitSet, sole(inputs));
	}

	@Override
	public Class<Object[]> getElementType() {
		return Object[].class;
	}

	@Override
	public Enumerable<Object[]> bind(DataContext dataContext) {
		return new Interpreter(dataContext, getInput());
	}
}
