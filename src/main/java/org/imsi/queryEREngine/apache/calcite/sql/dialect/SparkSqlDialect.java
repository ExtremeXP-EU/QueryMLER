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
package org.imsi.queryEREngine.apache.calcite.sql.dialect;

import static org.imsi.queryEREngine.apache.calcite.util.RelToSqlConverterUtil.unparseHiveTrim;

import org.apache.calcite.avatica.util.TimeUnitRange;
import org.imsi.queryEREngine.apache.calcite.config.NullCollation;
import org.imsi.queryEREngine.apache.calcite.sql.JoinType;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlDialect;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlLiteral;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlUtil;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.fun.SqlFloorFunction;
import org.imsi.queryEREngine.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;

/**
 * A <code>SqlDialect</code> implementation for the APACHE SPARK database.
 */
public class SparkSqlDialect extends SqlDialect {
	public static final SqlDialect.Context DEFAULT_CONTEXT = SqlDialect.EMPTY_CONTEXT
			.withDatabaseProduct(SqlDialect.DatabaseProduct.SPARK)
			.withNullCollation(NullCollation.LOW);

	public static final SqlDialect DEFAULT = new SparkSqlDialect(DEFAULT_CONTEXT);

	private static final SqlFunction SPARKSQL_SUBSTRING =
			new SqlFunction("SUBSTRING", SqlKind.OTHER_FUNCTION,
					ReturnTypes.ARG0_NULLABLE_VARYING, null, null,
					SqlFunctionCategory.STRING);

	/**
	 * Creates a SparkSqlDialect.
	 */
	public SparkSqlDialect(SqlDialect.Context context) {
		super(context);
	}

	@Override protected boolean allowsAs() {
		return false;
	}

	@Override public boolean supportsCharSet() {
		return false;
	}

	@Override public JoinType emulateJoinTypeForCrossJoin() {
		return JoinType.CROSS;
	}

	@Override public boolean supportsGroupByWithRollup() {
		return true;
	}

	@Override public boolean supportsNestedAggregations() {
		return false;
	}

	@Override public boolean supportsGroupByWithCube() {
		return true;
	}

	@Override public void unparseOffsetFetch(SqlWriter writer, SqlNode offset,
			SqlNode fetch) {
		unparseFetchUsingLimit(writer, offset, fetch);
	}

	@Override public void unparseCall(SqlWriter writer, SqlCall call,
			int leftPrec, int rightPrec) {
		if (call.getOperator() == SqlStdOperatorTable.SUBSTRING) {
			SqlUtil.unparseFunctionSyntax(SPARKSQL_SUBSTRING, writer, call);
		} else {
			switch (call.getKind()) {
			case FLOOR:
				if (call.operandCount() != 2) {
					super.unparseCall(writer, call, leftPrec, rightPrec);
					return;
				}

				final SqlLiteral timeUnitNode = call.operand(1);
				final TimeUnitRange timeUnit = timeUnitNode.getValueAs(TimeUnitRange.class);

				SqlCall call2 = SqlFloorFunction.replaceTimeUnitOperand(call, timeUnit.name(),
						timeUnitNode.getParserPosition());
				SqlFloorFunction.unparseDatetimeFunction(writer, call2, "DATE_TRUNC", false);
				break;
			case TRIM:
				unparseHiveTrim(writer, call, leftPrec, rightPrec);
				break;
			default:
				super.unparseCall(writer, call, leftPrec, rightPrec);
			}
		}
	}
}
