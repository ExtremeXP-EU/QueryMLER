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

import java.util.Objects;

import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.sql.SqlAggFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlJsonConstructorNullClause;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlLiteral;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.InferTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeFamily;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorImpl;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorScope;
import org.imsi.queryEREngine.apache.calcite.util.Optionality;

/**
 * The <code>JSON_OBJECTAGG</code> aggregate function.
 */
public class SqlJsonArrayAggAggFunction extends SqlAggFunction {
	private final SqlJsonConstructorNullClause nullClause;

	public SqlJsonArrayAggAggFunction(SqlKind kind,
			SqlJsonConstructorNullClause nullClause) {
		super(kind + "_" + nullClause.name(), null, kind, ReturnTypes.VARCHAR_2000,
				InferTypes.ANY_NULLABLE, OperandTypes.family(SqlTypeFamily.ANY),
				SqlFunctionCategory.SYSTEM, false, false, Optionality.OPTIONAL);
		this.nullClause = Objects.requireNonNull(nullClause);
	}

	@Override public void unparse(SqlWriter writer, SqlCall call, int leftPrec,
			int rightPrec) {
		assert call.operandCount() == 1;
		final SqlWriter.Frame frame = writer.startFunCall("JSON_ARRAYAGG");
		call.operand(0).unparse(writer, leftPrec, rightPrec);
		writer.keyword(nullClause.sql);
		writer.endFunCall(frame);
	}

	@Override public RelDataType deriveType(SqlValidator validator,
			SqlValidatorScope scope, SqlCall call) {
		// To prevent operator rewriting by SqlFunction#deriveType.
		for (SqlNode operand : call.getOperandList()) {
			RelDataType nodeType = validator.deriveType(scope, operand);
			((SqlValidatorImpl) validator).setValidatedNodeType(operand, nodeType);
		}
		return validateOperands(validator, scope, call);
	}

	@Override public SqlCall createCall(SqlLiteral functionQualifier,
			SqlParserPos pos, SqlNode... operands) {
		assert operands.length == 1 || operands.length == 2;
		final SqlNode valueExpr = operands[0];
		if (operands.length == 2) {
			final SqlNode orderList = operands[1];
			if (orderList != null) {
				// call has an order by clause, e.g. json_arrayagg(col_1 order by col_1)
				return SqlStdOperatorTable.WITHIN_GROUP.createCall(SqlParserPos.ZERO,
						createCall_(functionQualifier, pos, valueExpr), orderList);
			}
		}
		return createCall_(functionQualifier, pos, valueExpr);
	}

	private SqlCall createCall_(SqlLiteral functionQualifier, SqlParserPos pos, SqlNode valueExpr) {
		return super.createCall(functionQualifier, pos, valueExpr);
	}

	public SqlJsonArrayAggAggFunction with(SqlJsonConstructorNullClause nullClause) {
		return this.nullClause == nullClause ? this
				: new SqlJsonArrayAggAggFunction(getKind(), nullClause);
	}

	public SqlJsonConstructorNullClause getNullClause() {
		return nullClause;
	}
}
