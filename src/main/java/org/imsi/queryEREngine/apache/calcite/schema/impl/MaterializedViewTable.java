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
package org.imsi.queryEREngine.apache.calcite.schema.impl;

import java.lang.reflect.Type;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.List;
import java.util.Objects;

import org.imsi.queryEREngine.apache.calcite.adapter.java.JavaTypeFactory;
import org.imsi.queryEREngine.apache.calcite.jdbc.CalciteConnection;
import org.imsi.queryEREngine.apache.calcite.jdbc.CalcitePrepare;
import org.imsi.queryEREngine.apache.calcite.jdbc.CalciteSchema;
import org.imsi.queryEREngine.apache.calcite.materialize.MaterializationKey;
import org.imsi.queryEREngine.apache.calcite.materialize.MaterializationService;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeImpl;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelProtoDataType;
import org.imsi.queryEREngine.apache.calcite.schema.Schemas;
import org.imsi.queryEREngine.apache.calcite.schema.Table;
import org.imsi.queryEREngine.apache.calcite.schema.TranslatableTable;

/**
 * Table that is a materialized view.
 *
 * <p>It can exist in two states: materialized and not materialized. Over time,
 * a given materialized view may switch states. How it is expanded depends upon
 * its current state. State is managed by
 * {@link org.imsi.queryEREngine.apache.calcite.materialize.MaterializationService}.</p>
 */
public class MaterializedViewTable extends ViewTable {

	private final MaterializationKey key;

	/**
	 * Internal connection, used to execute queries to materialize views.
	 * To be used only by Calcite internals. And sparingly.
	 */
	public static final CalciteConnection MATERIALIZATION_CONNECTION;

	static {
		try {
			MATERIALIZATION_CONNECTION = DriverManager.getConnection("jdbc:calcite:")
					.unwrap(CalciteConnection.class);
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}

	public MaterializedViewTable(Type elementType,
			RelProtoDataType relDataType,
			String viewSql,
			List<String> viewSchemaPath,
			List<String> viewPath,
			MaterializationKey key) {
		super(elementType, relDataType, viewSql, viewSchemaPath, viewPath);
		this.key = key;
	}

	/** Table macro that returns a materialized view. */
	public static MaterializedViewTableMacro create(final CalciteSchema schema,
			final String viewSql, final List<String> viewSchemaPath, List<String> viewPath,
			final String suggestedTableName, boolean existing) {
		return new MaterializedViewTableMacro(schema, viewSql, viewSchemaPath, viewPath,
				suggestedTableName, existing);
	}

	@Override public RelNode toRel(RelOptTable.ToRelContext context,
			RelOptTable relOptTable) {
		final CalciteSchema.TableEntry tableEntry =
				MaterializationService.instance().checkValid(key);
		if (tableEntry != null) {
			Table materializeTable = tableEntry.getTable();
			if (materializeTable instanceof TranslatableTable) {
				TranslatableTable table = (TranslatableTable) materializeTable;
				return table.toRel(context, relOptTable);
			}
		}
		return super.toRel(context, relOptTable);
	}

	/** Table function that returns the table that materializes a view. */
	public static class MaterializedViewTableMacro
	extends ViewTableMacro {
		private final MaterializationKey key;

		private MaterializedViewTableMacro(CalciteSchema schema, String viewSql,
				List<String> viewSchemaPath, List<String> viewPath, String suggestedTableName,
				boolean existing) {
			super(schema, viewSql,
					viewSchemaPath != null ? viewSchemaPath : schema.path(null), viewPath,
							Boolean.TRUE);
			this.key = Objects.requireNonNull(
					MaterializationService.instance().defineMaterialization(
							schema, null, viewSql, schemaPath, suggestedTableName, true,
							existing));
		}

		@Override public TranslatableTable apply(List<Object> arguments) {
			assert arguments.isEmpty();
			CalcitePrepare.ParseResult parsed =
					Schemas.parse(MATERIALIZATION_CONNECTION, schema, schemaPath,
							viewSql);
			final List<String> schemaPath1 =
					schemaPath != null ? schemaPath : schema.path(null);
			final JavaTypeFactory typeFactory =
					MATERIALIZATION_CONNECTION.getTypeFactory();
			return new MaterializedViewTable(typeFactory.getJavaClass(parsed.rowType),
					RelDataTypeImpl.proto(parsed.rowType), viewSql, schemaPath1, viewPath, key);
		}
	}
}
