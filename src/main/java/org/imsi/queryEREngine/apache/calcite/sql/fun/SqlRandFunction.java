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

import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSyntax;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;

/**
 * The <code>RAND</code> function. There are two overloads:
 *
 * <ul>
 *   <li>RAND() returns a random double between 0 and 1
 *   <li>RAND(seed) returns a random double between 0 and 1, initializing the
 *   random number generator with seed on first call
 * </ul>
 */
public class SqlRandFunction extends SqlFunction {
	//~ Constructors -----------------------------------------------------------

	public SqlRandFunction() {
		super("RAND",
				SqlKind.OTHER_FUNCTION,
				ReturnTypes.DOUBLE,
				null,
				OperandTypes.or(OperandTypes.NILADIC, OperandTypes.NUMERIC),
				SqlFunctionCategory.NUMERIC);
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public SqlSyntax getSyntax() {
		return SqlSyntax.FUNCTION;
	}

	// Plans referencing context variables should never be cached
	@Override
	public boolean isDynamicFunction() {
		return true;
	}
}
