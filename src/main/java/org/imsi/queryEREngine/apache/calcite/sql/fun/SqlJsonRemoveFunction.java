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

import java.util.Locale;

import org.imsi.queryEREngine.apache.calcite.sql.SqlCallBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperandCountRange;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandCountRanges;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeFamily;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeTransforms;

/**
 * The <code>JSON_REMOVE</code> function.
 */
public class SqlJsonRemoveFunction extends SqlFunction {

	public SqlJsonRemoveFunction() {
		super("JSON_REMOVE",
				SqlKind.OTHER_FUNCTION,
				ReturnTypes.cascade(ReturnTypes.VARCHAR_2000,
						SqlTypeTransforms.FORCE_NULLABLE),
				null,
				null,
				SqlFunctionCategory.SYSTEM);
	}

	@Override public SqlOperandCountRange getOperandCountRange() {
		return SqlOperandCountRanges.from(2);
	}

	@Override public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
		final int operandCount = callBinding.getOperandCount();
		assert operandCount >= 2;
		if (!OperandTypes.ANY.checkSingleOperandType(
				callBinding, callBinding.operand(0), 0, throwOnFailure)) {
			return false;
		}
		final SqlTypeFamily[] families = new SqlTypeFamily[operandCount];
		families[0] = SqlTypeFamily.ANY;
		for (int i = 1; i < operandCount; i++) {
			families[i] = SqlTypeFamily.CHARACTER;
		}
		return OperandTypes.family(families).checkOperandTypes(callBinding, throwOnFailure);
	}

	@Override public String getAllowedSignatures(String opNameToUse) {
		return String.format(Locale.ROOT, "'%s(<%s>, <%s>, <%s>...)'", getName(), SqlTypeFamily.ANY,
				SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER);
	}
}
