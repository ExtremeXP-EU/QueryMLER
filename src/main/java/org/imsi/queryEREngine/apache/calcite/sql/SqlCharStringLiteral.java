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
package org.imsi.queryEREngine.apache.calcite.sql;

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.util.Bug;
import org.imsi.queryEREngine.apache.calcite.util.NlsString;
import org.imsi.queryEREngine.apache.calcite.util.Util;

/**
 * A character string literal.
 *
 * <p>Its {@link #value} field is an {@link NlsString} and
 * {@link #getTypeName typeName} is {@link SqlTypeName#CHAR}.
 */
public class SqlCharStringLiteral extends SqlAbstractStringLiteral {

	//~ Constructors -----------------------------------------------------------

	protected SqlCharStringLiteral(NlsString val, SqlParserPos pos) {
		super(val, SqlTypeName.CHAR, pos);
	}

	//~ Methods ----------------------------------------------------------------

	/**
	 * @return the underlying NlsString
	 */
	public NlsString getNlsString() {
		return (NlsString) value;
	}

	/**
	 * @return the collation
	 */
	public SqlCollation getCollation() {
		return getNlsString().getCollation();
	}

	@Override public SqlCharStringLiteral clone(SqlParserPos pos) {
		return new SqlCharStringLiteral((NlsString) value, pos);
	}

	@Override
	public void unparse(
			SqlWriter writer,
			int leftPrec,
			int rightPrec) {
		assert value instanceof NlsString;
		final NlsString nlsString = (NlsString) this.value;
		if (false) {
			Util.discard(Bug.FRG78_FIXED);
			String stringValue = nlsString.getValue();
			writer.literal(
					writer.getDialect().quoteStringLiteral(stringValue));
		}
		writer.literal(nlsString.asSql(true, true, writer.getDialect()));
	}

	@Override
	protected SqlAbstractStringLiteral concat1(List<SqlLiteral> literals) {
		return new SqlCharStringLiteral(
				NlsString.concat(
						Util.transform(literals,
								literal -> ((SqlCharStringLiteral) literal).getNlsString())),
				literals.get(0).getParserPosition());
	}
}
