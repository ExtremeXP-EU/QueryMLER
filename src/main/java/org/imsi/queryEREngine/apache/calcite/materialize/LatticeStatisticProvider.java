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
package org.imsi.queryEREngine.apache.calcite.materialize;

import java.util.List;
import java.util.function.Function;

/**
 * Estimates row counts for a lattice and its attributes.
 */
public interface LatticeStatisticProvider {
	/** Returns an estimate of the number of distinct values in a column
	 * or list of columns. */
	double cardinality(List<Lattice.Column> columns);

	/** Creates a {@link LatticeStatisticProvider} for a given
	 * {@link org.imsi.queryEREngine.apache.calcite.materialize.Lattice}. */
	interface Factory extends Function<Lattice, LatticeStatisticProvider> {
	}
}
