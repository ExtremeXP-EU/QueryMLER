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

/**
 * Information for a call to
 * {@link AggImplementor#implementReset(AggContext, AggResetContext)}.
 *
 * <p>The {@link AggResetContext} provides access to the accumulator variables
 * that should be reset.
 *
 * <p>Note: the very first reset of windowed aggregates is performed with null
 * knowledge of indices and row count in the partition.
 * In other words, the implementation should treat indices and partition row
 * count as a hint to pre-size the collections.
 */
public interface WinAggResetContext
extends AggResetContext, WinAggFrameContext {
}
