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
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperatorBinding;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandTypeInference;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlReturnTypeInference;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlMonotonicity;

/**
 * Base class for unary operators such as FLOOR/CEIL which are monotonic for
 * monotonic inputs.
 */
public class SqlMonotonicUnaryFunction extends SqlFunction {
	//~ Constructors -----------------------------------------------------------

	protected SqlMonotonicUnaryFunction(
			String name,
			SqlKind kind,
			SqlReturnTypeInference returnTypeInference,
			SqlOperandTypeInference operandTypeInference,
			SqlOperandTypeChecker operandTypeChecker,
			SqlFunctionCategory funcType) {
		super(
				name,
				kind,
				returnTypeInference,
				operandTypeInference,
				operandTypeChecker,
				funcType);
	}

	//~ Methods ----------------------------------------------------------------

	@Override public SqlMonotonicity getMonotonicity(SqlOperatorBinding call) {
		return call.getOperandMonotonicity(0).unstrict();
	}
}
