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

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlCallBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperatorBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSelect;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSpecialOperator;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeUtil;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorNamespace;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorScope;

/**
 * Definition of the SQL:2003 standard MULTISET query constructor, <code>
 * MULTISET (&lt;query&gt;)</code>.
 *
 * @see SqlMultisetValueConstructor
 */
public class SqlMultisetQueryConstructor extends SqlSpecialOperator {
	//~ Constructors -----------------------------------------------------------

	public SqlMultisetQueryConstructor() {
		this("MULTISET", SqlKind.MULTISET_QUERY_CONSTRUCTOR);
	}

	protected SqlMultisetQueryConstructor(String name, SqlKind kind) {
		super(
				name,
				kind, MDX_PRECEDENCE,
				false,
				ReturnTypes.ARG0,
				null,
				OperandTypes.VARIADIC);
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public RelDataType inferReturnType(
			SqlOperatorBinding opBinding) {
		RelDataType type =
				getComponentType(
						opBinding.getTypeFactory(),
						opBinding.collectOperandTypes());
		if (null == type) {
			return null;
		}
		return SqlTypeUtil.createMultisetType(
				opBinding.getTypeFactory(),
				type,
				false);
	}

	private RelDataType getComponentType(
			RelDataTypeFactory typeFactory,
			List<RelDataType> argTypes) {
		return typeFactory.leastRestrictive(argTypes);
	}

	@Override
	public boolean checkOperandTypes(
			SqlCallBinding callBinding,
			boolean throwOnFailure) {
		final List<RelDataType> argTypes =
				SqlTypeUtil.deriveAndCollectTypes(
						callBinding.getValidator(),
						callBinding.getScope(),
						callBinding.operands());
		final RelDataType componentType =
				getComponentType(
						callBinding.getTypeFactory(),
						argTypes);
		if (null == componentType) {
			if (throwOnFailure) {
				throw callBinding.newValidationError(RESOURCE.needSameTypeParameter());
			}
			return false;
		}
		return true;
	}

	@Override
	public RelDataType deriveType(
			SqlValidator validator,
			SqlValidatorScope scope,
			SqlCall call) {
		SqlSelect subSelect = call.operand(0);
		subSelect.validateExpr(validator, scope);
		SqlValidatorNamespace ns = validator.getNamespace(subSelect);
		assert null != ns.getRowType();
		return SqlTypeUtil.createMultisetType(
				validator.getTypeFactory(),
				ns.getRowType(),
				false);
	}

	@Override
	public void unparse(
			SqlWriter writer,
			SqlCall call,
			int leftPrec,
			int rightPrec) {
		writer.keyword(getName());
		final SqlWriter.Frame frame = writer.startList("(", ")");
		assert call.operandCount() == 1;
		call.operand(0).unparse(writer, leftPrec, rightPrec);
		writer.endList(frame);
	}

	@Override
	public boolean argumentMustBeScalar(int ordinal) {
		return false;
	}
}
