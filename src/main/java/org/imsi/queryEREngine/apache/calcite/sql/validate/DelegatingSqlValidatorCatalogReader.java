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
import org.imsi.queryEREngine.apache.calcite.sql.SqlIdentifier;

/**
 * Implementation of
 * {@link org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidatorCatalogReader} that passes
 * all calls to a parent catalog reader.
 */
public abstract class DelegatingSqlValidatorCatalogReader
implements SqlValidatorCatalogReader {
	protected final SqlValidatorCatalogReader catalogReader;

	/**
	 * Creates a DelegatingSqlValidatorCatalogReader.
	 *
	 * @param catalogReader Parent catalog reader
	 */
	public DelegatingSqlValidatorCatalogReader(
			SqlValidatorCatalogReader catalogReader) {
		this.catalogReader = catalogReader;
	}

	@Override
	public SqlValidatorTable getTable(List<String> names) {
		return catalogReader.getTable(names);
	}

	@Override
	public RelDataType getNamedType(SqlIdentifier typeName) {
		return catalogReader.getNamedType(typeName);
	}

	@Override
	public List<SqlMoniker> getAllSchemaObjectNames(List<String> names) {
		return catalogReader.getAllSchemaObjectNames(names);
	}

	@Override
	public List<List<String>> getSchemaPaths() {
		return catalogReader.getSchemaPaths();
	}

	@Override public <C> C unwrap(Class<C> aClass) {
		return catalogReader.unwrap(aClass);
	}
}
