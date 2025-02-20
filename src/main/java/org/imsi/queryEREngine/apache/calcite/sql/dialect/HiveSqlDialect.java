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

import org.imsi.queryEREngine.apache.calcite.config.NullCollation;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.sql.SqlAlienSystemTypeNameSpec;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlDataTypeSpec;
import org.imsi.queryEREngine.apache.calcite.sql.SqlDialect;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperator;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSyntax;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.imsi.queryEREngine.apache.calcite.sql.fun.SqlSubstringFunction;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.BasicSqlType;
import org.imsi.queryEREngine.apache.calcite.util.RelToSqlConverterUtil;

/**
 * A <code>SqlDialect</code> implementation for the Apache Hive database.
 */
public class HiveSqlDialect extends SqlDialect {
	public static final SqlDialect.Context DEFAULT_CONTEXT = SqlDialect.EMPTY_CONTEXT
			.withDatabaseProduct(SqlDialect.DatabaseProduct.HIVE)
			.withNullCollation(NullCollation.LOW);

	public static final SqlDialect DEFAULT = new HiveSqlDialect(DEFAULT_CONTEXT);

	private final boolean emulateNullDirection;

	/** Creates a HiveSqlDialect. */
	public HiveSqlDialect(Context context) {
		super(context);
		// Since 2.1.0, Hive natively supports "NULLS FIRST" and "NULLS LAST".
		// See https://issues.apache.org/jira/browse/HIVE-12994.
		emulateNullDirection = (context.databaseMajorVersion() < 2)
				|| (context.databaseMajorVersion() == 2
				&& context.databaseMinorVersion() < 1);
	}

	@Override protected boolean allowsAs() {
		return false;
	}

	@Override public void unparseOffsetFetch(SqlWriter writer, SqlNode offset,
			SqlNode fetch) {
		unparseFetchUsingLimit(writer, offset, fetch);
	}

	@Override public SqlNode emulateNullDirection(SqlNode node,
			boolean nullsFirst, boolean desc) {
		if (emulateNullDirection) {
			return emulateNullDirectionWithIsNull(node, nullsFirst, desc);
		}

		return null;
	}

	@Override public void unparseCall(final SqlWriter writer, final SqlCall call,
			final int leftPrec, final int rightPrec) {
		switch (call.getKind()) {
		case POSITION:
			final SqlWriter.Frame frame = writer.startFunCall("INSTR");
			writer.sep(",");
			call.operand(1).unparse(writer, leftPrec, rightPrec);
			writer.sep(",");
			call.operand(0).unparse(writer, leftPrec, rightPrec);
			if (3 == call.operandCount()) {
				throw new RuntimeException("3rd operand Not Supported for Function INSTR in Hive");
			}
			writer.endFunCall(frame);
			break;
		case MOD:
			SqlOperator op = SqlStdOperatorTable.PERCENT_REMAINDER;
			SqlSyntax.BINARY.unparse(writer, op, call, leftPrec, rightPrec);
			break;
		case TRIM:
			RelToSqlConverterUtil.unparseHiveTrim(writer, call, leftPrec, rightPrec);
			break;
		case OTHER_FUNCTION:
			if (call.getOperator() instanceof SqlSubstringFunction) {
				final SqlWriter.Frame funCallFrame = writer.startFunCall(call.getOperator().getName());
				call.operand(0).unparse(writer, leftPrec, rightPrec);
				writer.sep(",", true);
				call.operand(1).unparse(writer, leftPrec, rightPrec);
				if (3 == call.operandCount()) {
					writer.sep(",", true);
					call.operand(2).unparse(writer, leftPrec, rightPrec);
				}
				writer.endFunCall(funCallFrame);
			} else {
				super.unparseCall(writer, call, leftPrec, rightPrec);
			}
			break;
		default:
			super.unparseCall(writer, call, leftPrec, rightPrec);
		}
	}

	@Override public boolean supportsCharSet() {
		return false;
	}

	@Override public boolean supportsGroupByWithRollup() {
		return true;
	}

	@Override public boolean supportsGroupByWithCube() {
		return true;
	}

	@Override public boolean supportsNestedAggregations() {
		return false;
	}

	@Override public SqlNode getCastSpec(final RelDataType type) {
		if (type instanceof BasicSqlType) {
			switch (type.getSqlTypeName()) {
			case INTEGER:
				SqlAlienSystemTypeNameSpec typeNameSpec = new SqlAlienSystemTypeNameSpec(
						"INT", type.getSqlTypeName(), SqlParserPos.ZERO);
				return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
			}
		}
		return super.getCastSpec(type);
	}
}
