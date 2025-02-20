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

import java.util.List;

import org.apache.calcite.linq4j.tree.Expression;
import org.imsi.queryEREngine.apache.calcite.rel.core.AggregateCall;

/**
 * Represents internal state when implementing aggregate functions.
 */
public class AggImpState {
	public final int aggIdx;
	public final AggregateCall call;
	public final AggImplementor implementor;
	public AggContext context;
	public Expression result;
	public List<Expression> state;
	public Expression accumulatorAdder;

	public AggImpState(int aggIdx, AggregateCall call, boolean windowContext) {
		this.aggIdx = aggIdx;
		this.call = call;
		this.implementor =
				RexImpTable.INSTANCE.get(call.getAggregation(), windowContext);
		if (implementor == null) {
			throw new IllegalArgumentException(
					"Unable to get aggregate implementation for aggregate "
							+ call.getAggregation()
							+ (windowContext ? " in window context" : ""));
		}
	}
}
