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

import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlLiteral;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperandCountRange;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandCountRanges;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeTransforms;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;

/**
 * The <code>JSON_TYPE</code> function.
 */
public class SqlJsonPrettyFunction extends SqlFunction {

	public SqlJsonPrettyFunction() {
		super("JSON_PRETTY",
				SqlKind.OTHER_FUNCTION,
				ReturnTypes.cascade(ReturnTypes.VARCHAR_2000, SqlTypeTransforms.FORCE_NULLABLE),
				null,
				OperandTypes.ANY,
				SqlFunctionCategory.SYSTEM);
	}

	@Override public SqlOperandCountRange getOperandCountRange() {
		return SqlOperandCountRanges.of(1);
	}

	@Override protected void checkOperandCount(SqlValidator validator,
			SqlOperandTypeChecker argType, SqlCall call) {
		assert call.operandCount() == 1;
	}

	@Override public SqlCall createCall(SqlLiteral functionQualifier,
			SqlParserPos pos, SqlNode... operands) {
		return super.createCall(functionQualifier, pos, operands);
	}
}
