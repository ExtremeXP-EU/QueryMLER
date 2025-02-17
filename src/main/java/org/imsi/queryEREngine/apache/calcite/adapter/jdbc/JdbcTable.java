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
package org.imsi.queryEREngine.apache.calcite.adapter.jdbc;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

import org.imsi.queryEREngine.apache.calcite.DataContext;
import org.imsi.queryEREngine.apache.calcite.adapter.java.AbstractQueryableTable;
import org.imsi.queryEREngine.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.avatica.ColumnMetaData;
import org.imsi.queryEREngine.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.linq4j.QueryProvider;
import org.apache.calcite.linq4j.Queryable;
import org.imsi.queryEREngine.apache.calcite.plan.Convention;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.prepare.Prepare.CatalogReader;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableModify;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableModify.Operation;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableModify;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelProtoDataType;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.runtime.ResultSetEnumerable;
import org.imsi.queryEREngine.apache.calcite.schema.ModifiableTable;
import org.imsi.queryEREngine.apache.calcite.schema.ScannableTable;
import org.imsi.queryEREngine.apache.calcite.schema.Schema;
import org.imsi.queryEREngine.apache.calcite.schema.SchemaPlus;
import org.imsi.queryEREngine.apache.calcite.schema.TranslatableTable;
import org.imsi.queryEREngine.apache.calcite.schema.impl.AbstractTableQueryable;
import org.imsi.queryEREngine.apache.calcite.sql.SqlIdentifier;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNodeList;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSelect;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWriterConfig;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.imsi.queryEREngine.apache.calcite.sql.util.SqlString;
import org.imsi.queryEREngine.apache.calcite.util.Pair;
import org.imsi.queryEREngine.apache.calcite.util.Util;

import com.google.common.collect.Lists;

/**
 * Queryable that gets its data from a table within a JDBC connection.
 *
 * <p>The idea is not to read the whole table, however. The idea is to use
 * this as a building block for a query, by applying Queryable operators
 * such as
 * {@link org.apache.calcite.linq4j.Queryable#where(org.apache.calcite.linq4j.function.Predicate2)}.
 * The resulting queryable can then be converted to a SQL query, which can be
 * executed efficiently on the JDBC server.</p>
 */
public class JdbcTable extends AbstractQueryableTable
implements TranslatableTable, ScannableTable, ModifiableTable {
	private RelProtoDataType protoRowType;
	public final JdbcSchema jdbcSchema;
	public final String jdbcCatalogName;
	public final String jdbcSchemaName;
	public final String jdbcTableName;
	public final Schema.TableType jdbcTableType;

	JdbcTable(JdbcSchema jdbcSchema, String jdbcCatalogName,
			String jdbcSchemaName, String jdbcTableName,
			Schema.TableType jdbcTableType) {
		super(Object[].class);
		this.jdbcSchema = Objects.requireNonNull(jdbcSchema);
		this.jdbcCatalogName = jdbcCatalogName;
		this.jdbcSchemaName = jdbcSchemaName;
		this.jdbcTableName = Objects.requireNonNull(jdbcTableName);
		this.jdbcTableType = Objects.requireNonNull(jdbcTableType);
	}

	@Override
	public String toString() {
		return "JdbcTable {" + jdbcTableName + "}";
	}

	@Override public Schema.TableType getJdbcTableType() {
		return jdbcTableType;
	}

	@Override public <C> C unwrap(Class<C> aClass) {
		if (aClass.isInstance(jdbcSchema.getDataSource())) {
			return aClass.cast(jdbcSchema.getDataSource());
		} else if (aClass.isInstance(jdbcSchema.dialect)) {
			return aClass.cast(jdbcSchema.dialect);
		} else {
			return super.unwrap(aClass);
		}
	}

	@Override
	public RelDataType getRowType(RelDataTypeFactory typeFactory) {
		if (protoRowType == null) {
			try {
				protoRowType =
						jdbcSchema.getRelDataType(
								jdbcCatalogName,
								jdbcSchemaName,
								jdbcTableName);
			} catch (SQLException e) {
				throw new RuntimeException(
						"Exception while reading definition of table '" + jdbcTableName
						+ "'", e);
			}
		}
		return protoRowType.apply(typeFactory);
	}

	private List<Pair<ColumnMetaData.Rep, Integer>> fieldClasses(
			final JavaTypeFactory typeFactory) {
		final RelDataType rowType = protoRowType.apply(typeFactory);
		return Lists.transform(rowType.getFieldList(), f -> {
			final RelDataType type = f.getType();
			final Class clazz = (Class) typeFactory.getJavaClass(type);
			final ColumnMetaData.Rep rep =
					Util.first(ColumnMetaData.Rep.of(clazz),
							ColumnMetaData.Rep.OBJECT);
			return Pair.of(rep, type.getSqlTypeName().getJdbcOrdinal());
		});
	}

	SqlString generateSql() {
		final SqlNodeList selectList = SqlNodeList.SINGLETON_STAR;
		SqlSelect node =
				new SqlSelect(SqlParserPos.ZERO, SqlNodeList.EMPTY, selectList,
						tableName(), null, null, null, null, null, null, null, null);
		final SqlWriterConfig config = SqlPrettyWriter.config()
				.withAlwaysUseParentheses(true)
				.withDialect(jdbcSchema.dialect);
		final SqlPrettyWriter writer = new SqlPrettyWriter(config);
		node.unparse(writer, 0, 0);
		return writer.toSqlString();
	}

	/** Returns the table name, qualified with catalog and schema name if
	 * applicable, as a parse tree node ({@link SqlIdentifier}). */
	public SqlIdentifier tableName() {
		final List<String> names = new ArrayList<>(3);
		if (jdbcSchema.catalog != null) {
			names.add(jdbcSchema.catalog);
		}
		if (jdbcSchema.schema != null) {
			names.add(jdbcSchema.schema);
		}
		names.add(jdbcTableName);
		return new SqlIdentifier(names, SqlParserPos.ZERO);
	}

	@Override
	public RelNode toRel(RelOptTable.ToRelContext context,
			RelOptTable relOptTable) {
		return new JdbcTableScan(context.getCluster(), relOptTable, this,
				jdbcSchema.convention);
	}

	@Override
	public <T> Queryable<T> asQueryable(QueryProvider queryProvider,
			SchemaPlus schema, String tableName) {
		return new JdbcTableQueryable<>(queryProvider, schema, tableName);
	}

	@Override
	public Enumerable<Object[]> scan(DataContext root) {
		final JavaTypeFactory typeFactory = root.getTypeFactory();
		final SqlString sql = generateSql();
		return ResultSetEnumerable.of(jdbcSchema.getDataSource(), sql.getSql(),
				JdbcUtils.ObjectArrayRowBuilder.factory(fieldClasses(typeFactory)));
	}

	@Override public Collection getModifiableCollection() {
		return null;
	}

	@Override public TableModify toModificationRel(RelOptCluster cluster,
			RelOptTable table, CatalogReader catalogReader, RelNode input,
			Operation operation, List<String> updateColumnList,
			List<RexNode> sourceExpressionList, boolean flattened) {
		jdbcSchema.convention.register(cluster.getPlanner());

		return new LogicalTableModify(cluster, cluster.traitSetOf(Convention.NONE),
				table, catalogReader, input, operation, updateColumnList,
				sourceExpressionList, flattened);
	}

	/** Enumerable that returns the contents of a {@link JdbcTable} by connecting
	 * to the JDBC data source.
	 *
	 * @param <T> element type */
	private class JdbcTableQueryable<T> extends AbstractTableQueryable<T> {
		JdbcTableQueryable(QueryProvider queryProvider, SchemaPlus schema,
				String tableName) {
			super(queryProvider, schema, JdbcTable.this, tableName);
		}

		@Override public String toString() {
			return "JdbcTableQueryable {table: " + tableName + "}";
		}

		@Override
		public Enumerator<T> enumerator() {
			final JavaTypeFactory typeFactory =
					((CalciteConnection) queryProvider).getTypeFactory();
			final SqlString sql = generateSql();
			//noinspection unchecked
			final Enumerable<T> enumerable = (Enumerable<T>) ResultSetEnumerable.of(
					jdbcSchema.getDataSource(),
					sql.getSql(),
					JdbcUtils.ObjectArrayRowBuilder.factory(fieldClasses(typeFactory)));
			return enumerable.enumerator();
		}
	}
}
