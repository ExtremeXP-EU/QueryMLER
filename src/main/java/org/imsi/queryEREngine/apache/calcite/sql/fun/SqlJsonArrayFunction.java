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

import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlJsonConstructorNullClause;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlLiteral;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperandCountRange;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.InferTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandCountRanges;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;

/**
 * The <code>JSON_ARRAY</code> function.
 */
public class SqlJsonArrayFunction extends SqlFunction {
	public SqlJsonArrayFunction() {
		super("JSON_ARRAY", SqlKind.OTHER_FUNCTION, ReturnTypes.VARCHAR_2000,
				InferTypes.ANY_NULLABLE, OperandTypes.VARIADIC, SqlFunctionCategory.SYSTEM);
	}

	@Override public SqlOperandCountRange getOperandCountRange() {
		return SqlOperandCountRanges.from(1);
	}

	@Override protected void checkOperandCount(SqlValidator validator,
			SqlOperandTypeChecker argType, SqlCall call) {
		assert call.operandCount() >= 1;
	}

	@Override public SqlCall createCall(SqlLiteral functionQualifier,
			SqlParserPos pos, SqlNode... operands) {
		if (operands[0] == null) {
			operands[0] =
					SqlLiteral.createSymbol(SqlJsonConstructorNullClause.ABSENT_ON_NULL,
							pos);
		}
		return super.createCall(functionQualifier, pos, operands);
	}

	@Override public String getSignatureTemplate(int operandsCount) {
		assert operandsCount >= 1;
		final StringBuilder sb = new StringBuilder();
		sb.append("{0}(");
		for (int i = 1; i < operandsCount; i++) {
			sb.append(String.format(Locale.ROOT, "{%d} ", i + 1));
		}
		sb.append("{1})");
		return sb.toString();
	}

	@Override public void unparse(SqlWriter writer, SqlCall call, int leftPrec,
			int rightPrec) {
		assert call.operandCount() >= 1;
		final SqlWriter.Frame frame = writer.startFunCall(getName());
		SqlWriter.Frame listFrame = writer.startList("", "");
		for (int i = 1; i < call.operandCount(); i++) {
			writer.sep(",");
			call.operand(i).unparse(writer, leftPrec, rightPrec);
		}
		writer.endList(listFrame);

		SqlJsonConstructorNullClause nullClause = getEnumValue(call.operand(0));
		switch (nullClause) {
		case ABSENT_ON_NULL:
			writer.keyword("ABSENT ON NULL");
			break;
		case NULL_ON_NULL:
			writer.keyword("NULL ON NULL");
			break;
		default:
			throw new IllegalStateException("unreachable code");
		}
		writer.endFunCall(frame);
	}

	private <E extends Enum<E>> E getEnumValue(SqlNode operand) {
		return (E) ((SqlLiteral) operand).getValue();
	}
}
