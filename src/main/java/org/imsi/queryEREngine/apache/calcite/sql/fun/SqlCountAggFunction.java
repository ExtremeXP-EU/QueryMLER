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

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.config.CalciteSystemProperty;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlAggFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSplittableAggFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSyntax;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorScope;
import org.imsi.queryEREngine.apache.calcite.util.Optionality;

import com.google.common.collect.ImmutableList;

/**
 * Definition of the SQL <code>COUNT</code> aggregation function.
 *
 * <p><code>COUNT</code> is an aggregator which returns the number of rows which
 * have gone into it. With one argument (or more), it returns the number of rows
 * for which that argument (or all) is not <code>null</code>.
 */
public class SqlCountAggFunction extends SqlAggFunction {
	//~ Constructors -----------------------------------------------------------

	public SqlCountAggFunction(String name) {
		this(name, CalciteSystemProperty.STRICT.value() ? OperandTypes.ANY : OperandTypes.ONE_OR_MORE);
	}

	public SqlCountAggFunction(String name,
			SqlOperandTypeChecker sqlOperandTypeChecker) {
		super(name, null, SqlKind.COUNT, ReturnTypes.BIGINT, null,
				sqlOperandTypeChecker, SqlFunctionCategory.NUMERIC, false, false,
				Optionality.FORBIDDEN);
	}

	//~ Methods ----------------------------------------------------------------

	@Override public SqlSyntax getSyntax() {
		return SqlSyntax.FUNCTION_STAR;
	}

	@Override
	@SuppressWarnings("deprecation")
	public List<RelDataType> getParameterTypes(RelDataTypeFactory typeFactory) {
		return ImmutableList.of(
				typeFactory.createTypeWithNullability(
						typeFactory.createSqlType(SqlTypeName.ANY), true));
	}

	@Override
	@SuppressWarnings("deprecation")
	public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
		return typeFactory.createSqlType(SqlTypeName.BIGINT);
	}

	@Override
	public RelDataType deriveType(
			SqlValidator validator,
			SqlValidatorScope scope,
			SqlCall call) {
		// Check for COUNT(*) function.  If it is we don't
		// want to try and derive the "*"
		if (call.isCountStar()) {
			return validator.getTypeFactory().createSqlType(
					SqlTypeName.BIGINT);
		}
		return super.deriveType(validator, scope, call);
	}

	@Override public <T> T unwrap(Class<T> clazz) {
		if (clazz == SqlSplittableAggFunction.class) {
			return clazz.cast(SqlSplittableAggFunction.CountSplitter.INSTANCE);
		}
		return super.unwrap(clazz);
	}
}
