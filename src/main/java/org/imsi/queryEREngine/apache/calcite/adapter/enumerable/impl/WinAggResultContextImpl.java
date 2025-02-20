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
package org.imsi.queryEREngine.apache.calcite.adapter.enumerable.impl;

import java.util.List;
import java.util.function.Function;

import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.RexToLixTranslator;
import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.WinAggFrameResultContext;
import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.WinAggImplementor;
import org.imsi.queryEREngine.apache.calcite.adapter.enumerable.WinAggResultContext;
import org.apache.calcite.linq4j.tree.BlockBuilder;
import org.apache.calcite.linq4j.tree.Expression;

/**
 * Implementation of
 * {@link org.imsi.queryEREngine.apache.calcite.adapter.enumerable.WinAggResultContext}.
 */
public abstract class WinAggResultContextImpl extends AggResultContextImpl
implements WinAggResultContext {

	private final Function<BlockBuilder, WinAggFrameResultContext> frame;

	/**
	 * Creates window aggregate result context.
	 *
	 * @param block code block that will contain the added initialization
	 * @param accumulator accumulator variables that store the intermediate
	 *                    aggregate state
	 */
	public WinAggResultContextImpl(BlockBuilder block,
			List<Expression> accumulator,
			Function<BlockBuilder, WinAggFrameResultContext> frameContextBuilder) {
		super(block, null, accumulator, null, null);
		this.frame = frameContextBuilder;
	}

	@SuppressWarnings("Guava")
	@Deprecated // to be removed before 2.0
	public WinAggResultContextImpl(BlockBuilder block,
			List<Expression> accumulator,
			com.google.common.base.Function<BlockBuilder, WinAggFrameResultContext> frameContextBuilder) {
		this(block, accumulator,
				(Function<BlockBuilder, WinAggFrameResultContext>) frameContextBuilder::apply);
	}

	private WinAggFrameResultContext getFrame() {
		return frame.apply(currentBlock());
	}

	@Override
	public final List<Expression> arguments(Expression rowIndex) {
		return rowTranslator(rowIndex).translateList(rexArguments());
	}

	@Override
	public Expression computeIndex(Expression offset,
			WinAggImplementor.SeekType seekType) {
		return getFrame().computeIndex(offset, seekType);
	}

	@Override
	public Expression rowInFrame(Expression rowIndex) {
		return getFrame().rowInFrame(rowIndex);
	}

	@Override
	public Expression rowInPartition(Expression rowIndex) {
		return getFrame().rowInPartition(rowIndex);
	}

	@Override
	public RexToLixTranslator rowTranslator(Expression rowIndex) {
		return getFrame().rowTranslator(rowIndex)
				.setNullable(currentNullables());
	}

	@Override
	public Expression compareRows(Expression a, Expression b) {
		return getFrame().compareRows(a, b);
	}

	@Override
	public Expression index() {
		return getFrame().index();
	}

	@Override
	public Expression startIndex() {
		return getFrame().startIndex();
	}

	@Override
	public Expression endIndex() {
		return getFrame().endIndex();
	}

	@Override
	public Expression hasRows() {
		return getFrame().hasRows();
	}

	@Override
	public Expression getFrameRowCount() {
		return getFrame().getFrameRowCount();
	}

	@Override
	public Expression getPartitionRowCount() {
		return getFrame().getPartitionRowCount();
	}
}
