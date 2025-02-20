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

import static org.imsi.queryEREngine.apache.calcite.util.Static.RESOURCE;

import java.util.Locale;

import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCallBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlJsonConstructorNullClause;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlLiteral;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperandCountRange;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandCountRanges;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeUtil;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;

/**
 * The <code>JSON_OBJECT</code> function.
 */
public class SqlJsonObjectFunction extends SqlFunction {
	public SqlJsonObjectFunction() {
		super("JSON_OBJECT", SqlKind.OTHER_FUNCTION, ReturnTypes.VARCHAR_2000,
				(callBinding, returnType, operandTypes) -> {
					RelDataTypeFactory typeFactory = callBinding.getTypeFactory();
					for (int i = 0; i < operandTypes.length; i++) {
						if (i % 2 == 0) {
							operandTypes[i] = typeFactory.createSqlType(SqlTypeName.VARCHAR);
							continue;
						}
						operandTypes[i] = typeFactory.createTypeWithNullability(
								typeFactory.createSqlType(SqlTypeName.ANY), true);
					}
				}, null, SqlFunctionCategory.SYSTEM);
	}

	@Override public SqlOperandCountRange getOperandCountRange() {
		return SqlOperandCountRanges.from(1);
	}

	@Override protected void checkOperandCount(SqlValidator validator,
			SqlOperandTypeChecker argType, SqlCall call) {
		assert call.operandCount() % 2 == 1;
	}

	@Override public boolean checkOperandTypes(SqlCallBinding callBinding,
			boolean throwOnFailure) {
		final int count = callBinding.getOperandCount();
		for (int i = 1; i < count; i += 2) {
			RelDataType nameType = callBinding.getOperandType(i);
			if (!SqlTypeUtil.inCharFamily(nameType)) {
				if (throwOnFailure) {
					throw callBinding.newError(RESOURCE.expectedCharacter());
				}
				return false;
			}
			if (nameType.isNullable()) {
				if (throwOnFailure) {
					throw callBinding.newError(
							RESOURCE.argumentMustNotBeNull(
									callBinding.operand(i).toString()));
				}
				return false;
			}
		}
		return true;
	}

	@Override public SqlCall createCall(SqlLiteral functionQualifier,
			SqlParserPos pos, SqlNode... operands) {
		if (operands[0] == null) {
			operands[0] = SqlLiteral.createSymbol(
					SqlJsonConstructorNullClause.NULL_ON_NULL, pos);
		}
		return super.createCall(functionQualifier, pos, operands);
	}

	@Override public String getSignatureTemplate(int operandsCount) {
		assert operandsCount % 2 == 1;
		StringBuilder sb = new StringBuilder();
		sb.append("{0}(");
		for (int i = 1; i < operandsCount; i++) {
			sb.append(String.format(Locale.ROOT, "{%d} ", i + 1));
		}
		sb.append("{1})");
		return sb.toString();
	}

	@Override public void unparse(SqlWriter writer, SqlCall call, int leftPrec,
			int rightPrec) {
		assert call.operandCount() % 2 == 1;
		final SqlWriter.Frame frame = writer.startFunCall(getName());
		SqlWriter.Frame listFrame = writer.startList("", "");
		for (int i = 1; i < call.operandCount(); i += 2) {
			writer.sep(",");
			writer.keyword("KEY");
			call.operand(i).unparse(writer, leftPrec, rightPrec);
			writer.keyword("VALUE");
			call.operand(i + 1).unparse(writer, leftPrec, rightPrec);
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
