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
package org.imsi.queryEREngine.apache.calcite.schema.impl;

import org.imsi.queryEREngine.apache.calcite.config.CalciteConnectionConfig;
import org.imsi.queryEREngine.apache.calcite.schema.Schema;
import org.imsi.queryEREngine.apache.calcite.schema.Statistic;
import org.imsi.queryEREngine.apache.calcite.schema.Statistics;
import org.imsi.queryEREngine.apache.calcite.schema.Table;
import org.imsi.queryEREngine.apache.calcite.schema.Wrapper;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;

/**
 * Abstract base class for implementing {@link Table}.
 *
 * <p>Sub-classes should override {@link #isRolledUp} and
 * {@link Table#rolledUpColumnValidInsideAgg(String, SqlCall, SqlNode, CalciteConnectionConfig)}
 * if their table can potentially contain rolled up values. This information is
 * used by the validator to check for illegal uses of these columns.
 */
public abstract class AbstractTable implements Table, Wrapper {
	protected AbstractTable() {
	}

	// Default implementation. Override if you have statistics.
	@Override
	public Statistic getStatistic() {
		return Statistics.UNKNOWN;
	}

	@Override
	public Schema.TableType getJdbcTableType() {
		return Schema.TableType.TABLE;
	}

	@Override
	public <C> C unwrap(Class<C> aClass) {
		if (aClass.isInstance(this)) {
			return aClass.cast(this);
		}
		return null;
	}

	@Override public boolean isRolledUp(String column) {
		return false;
	}

	@Override public boolean rolledUpColumnValidInsideAgg(String column,
			SqlCall call, SqlNode parent, CalciteConnectionConfig config) {
		return true;
	}
}
