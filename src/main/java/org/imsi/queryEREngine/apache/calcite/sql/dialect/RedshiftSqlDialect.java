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

import org.apache.calcite.avatica.util.Casing;
import org.imsi.queryEREngine.apache.calcite.sql.SqlDialect;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriter;

/**
 * A <code>SqlDialect</code> implementation for the Redshift database.
 */
public class RedshiftSqlDialect extends SqlDialect {
	public static final SqlDialect.Context DEFAULT_CONTEXT = SqlDialect.EMPTY_CONTEXT
			.withDatabaseProduct(SqlDialect.DatabaseProduct.REDSHIFT)
			.withIdentifierQuoteString("\"")
			.withQuotedCasing(Casing.TO_LOWER)
			.withUnquotedCasing(Casing.TO_LOWER)
			.withCaseSensitive(false);

	public static final SqlDialect DEFAULT = new RedshiftSqlDialect(DEFAULT_CONTEXT);

	/** Creates a RedshiftSqlDialect. */
	public RedshiftSqlDialect(Context context) {
		super(context);
	}

	@Override public void unparseOffsetFetch(SqlWriter writer, SqlNode offset,
			SqlNode fetch) {
		unparseFetchUsingLimit(writer, offset, fetch);
	}
}
