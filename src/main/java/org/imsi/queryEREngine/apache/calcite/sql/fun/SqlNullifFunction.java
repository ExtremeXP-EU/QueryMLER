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

import org.imsi.queryEREngine.apache.calcite.sql.SqlCall;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunctionCategory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlLiteral;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNodeList;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;

/**
 * The <code>NULLIF</code> function.
 */
public class SqlNullifFunction extends SqlFunction {
	//~ Constructors -----------------------------------------------------------

	public SqlNullifFunction() {
		// NOTE jvs 26-July-2006:  We fill in the type strategies here,
		// but normally they are not used because the validator invokes
		// rewriteCall to convert NULLIF into CASE early.  However,
		// validator rewrite can optionally be disabled, in which case these
		// strategies are used.
		super("NULLIF",
				SqlKind.NULLIF,
				ReturnTypes.ARG0_FORCE_NULLABLE,
				null,
				OperandTypes.COMPARABLE_UNORDERED_COMPARABLE_UNORDERED,
				SqlFunctionCategory.SYSTEM);
	}

	//~ Methods ----------------------------------------------------------------

	// override SqlOperator
	@Override
	public SqlNode rewriteCall(SqlValidator validator, SqlCall call) {
		List<SqlNode> operands = call.getOperandList();
		SqlParserPos pos = call.getParserPosition();

		checkOperandCount(
				validator,
				getOperandTypeChecker(),
				call);
		assert operands.size() == 2;

		SqlNodeList whenList = new SqlNodeList(pos);
		SqlNodeList thenList = new SqlNodeList(pos);
		whenList.add(operands.get(1));
		thenList.add(SqlLiteral.createNull(SqlParserPos.ZERO));
		return SqlCase.createSwitched(pos, operands.get(0), whenList, thenList,
				SqlNode.clone(operands.get(0)));
	}
}
