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
package org.imsi.queryEREngine.apache.calcite.sql.validate;

import java.util.List;

import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.sql.SqlAccessType;

/**
 * Implements {@link org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorTable} by
 * delegating to a parent table.
 */
public abstract class DelegatingSqlValidatorTable implements SqlValidatorTable {
	protected final SqlValidatorTable table;

	/**
	 * Creates a DelegatingSqlValidatorTable.
	 *
	 * @param table Parent table
	 */
	public DelegatingSqlValidatorTable(SqlValidatorTable table) {
		this.table = table;
	}

	@Override
	public RelDataType getRowType() {
		return table.getRowType();
	}

	@Override
	public List<String> getQualifiedName() {
		return table.getQualifiedName();
	}

	@Override
	public SqlMonotonicity getMonotonicity(String columnName) {
		return table.getMonotonicity(columnName);
	}

	@Override
	public SqlAccessType getAllowedAccess() {
		return table.getAllowedAccess();
	}
}
