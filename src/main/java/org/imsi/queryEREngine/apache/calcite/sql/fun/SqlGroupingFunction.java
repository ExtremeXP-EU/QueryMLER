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
 * The {@code GROUPING} function.
 *
 * <p>Accepts 1 or more arguments.
 * Example: {@code GROUPING(deptno, gender)} returns
 * 3 if both deptno and gender are being grouped,
 * 2 if only deptno is being grouped,
 * 1 if only gender is being groped,
 * 0 if neither deptno nor gender are being grouped.
 *
 * <p>This function is defined in the SQL standard.
 * {@code GROUPING_ID} is a non-standard synonym.
 *
 * <p>Some examples are in {@code agg.iq}.
 */
class SqlGroupingFunction extends SqlAbstractGroupFunction {
	SqlGroupingFunction(String name) {
		super(name, SqlKind.GROUPING, ReturnTypes.BIGINT, null,
				OperandTypes.ONE_OR_MORE, SqlFunctionCategory.SYSTEM);
	}
}
