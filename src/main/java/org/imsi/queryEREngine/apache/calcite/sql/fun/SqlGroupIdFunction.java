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

import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;

/**
 * The {@code GROUP_ID()} function.
 *
 * <p>Accepts no arguments. The {@code GROUP_ID()} function distinguishes duplicate groups
 * resulting from a GROUP BY specification. It is useful in filtering out duplicate groupings
 * from the query result.
 *
 * <p>This function is not defined in the SQL standard; our implementation is
 * consistent with Oracle.
 *
 * <p>If n duplicates exist for a particular grouping, then {@code GROUP_ID()} function returns
 * numbers in the range 0 to n-1.
 *
 * <p>Some examples are in {@code agg.iq}.
 */
class SqlGroupIdFunction extends SqlAbstractGroupFunction {
	SqlGroupIdFunction() {
		super("GROUP_ID", SqlKind.GROUP_ID, ReturnTypes.BIGINT, null,
				OperandTypes.NILADIC, SqlFunctionCategory.SYSTEM);
	}
}
