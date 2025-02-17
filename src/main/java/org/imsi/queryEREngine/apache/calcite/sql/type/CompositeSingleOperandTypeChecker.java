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
package org.imsi.queryEREngine.apache.calcite.sql.type;

import org.imsi.queryEREngine.apache.calcite.sql.SqlCallBinding;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;

/**
 * Allows multiple
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.SqlSingleOperandTypeChecker} rules to be
 * combined into one rule.
 */
public class CompositeSingleOperandTypeChecker
extends CompositeOperandTypeChecker
implements SqlSingleOperandTypeChecker {

	//~ Constructors -----------------------------------------------------------

	/**
	 * Package private. Use {@link org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes#and},
	 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.OperandTypes#or}.
	 */
	CompositeSingleOperandTypeChecker(
			CompositeOperandTypeChecker.Composition composition,
			ImmutableList<? extends SqlSingleOperandTypeChecker> allowedRules,
			String allowedSignatures) {
		super(composition, allowedRules, allowedSignatures, null);
	}

	//~ Methods ----------------------------------------------------------------

	@SuppressWarnings("unchecked")
	@Override public ImmutableList<? extends SqlSingleOperandTypeChecker>
	getRules() {
		return (ImmutableList<? extends SqlSingleOperandTypeChecker>) allowedRules;
	}

	@Override
	public boolean checkSingleOperandType(
			SqlCallBinding callBinding,
			SqlNode node,
			int iFormalOperand,
			boolean throwOnFailure) {
		assert allowedRules.size() >= 1;

		final ImmutableList<? extends SqlSingleOperandTypeChecker> rules =
				getRules();
		if (composition == Composition.SEQUENCE) {
			return rules.get(iFormalOperand).checkSingleOperandType(
					callBinding, node, 0, throwOnFailure);
		}

		int typeErrorCount = 0;

		boolean throwOnAndFailure =
				(composition == Composition.AND)
				&& throwOnFailure;

		for (SqlSingleOperandTypeChecker rule : rules) {
			if (!rule.checkSingleOperandType(
					callBinding,
					node,
					iFormalOperand,
					throwOnAndFailure)) {
				typeErrorCount++;
			}
		}

		boolean ret;
		switch (composition) {
		case AND:
			ret = typeErrorCount == 0;
			break;
		case OR:
			ret = typeErrorCount < allowedRules.size();
			break;
		default:
			// should never come here
			throw Util.unexpected(composition);
		}

		if (!ret && throwOnFailure) {
			// In the case of a composite OR, we want to throw an error
			// describing in more detail what the problem was, hence doing the
			// loop again.
			for (SqlSingleOperandTypeChecker rule : rules) {
				rule.checkSingleOperandType(
						callBinding,
						node,
						iFormalOperand,
						true);
			}

			// If no exception thrown, just throw a generic validation signature
			// error.
			throw callBinding.newValidationSignatureError();
		}

		return ret;
	}
}
