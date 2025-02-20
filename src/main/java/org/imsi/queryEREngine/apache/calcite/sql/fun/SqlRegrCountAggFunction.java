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
package org.imsi.queryEREngine.apache.calcite.sql.fun;

import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;

import com.google.common.base.Preconditions;

/**
 * Definition of the SQL <code>REGR_COUNT</code> aggregation function.
 *
 * <p><code>REGR_COUNT</code> is an aggregator which returns the number of rows which
 * have gone into it and both arguments are not <code>null</code>.
 */
public class SqlRegrCountAggFunction extends SqlCountAggFunction {
	public SqlRegrCountAggFunction(SqlKind kind) {
		super("REGR_COUNT", OperandTypes.NUMERIC_NUMERIC);
		Preconditions.checkArgument(SqlKind.REGR_COUNT == kind, "unsupported sql kind: " + kind);
	}
}
