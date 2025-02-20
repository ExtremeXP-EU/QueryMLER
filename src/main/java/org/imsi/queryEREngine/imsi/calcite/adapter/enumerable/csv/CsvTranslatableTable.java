/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.	See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.	You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.imsi.queryEREngine.imsi.calcite.adapter.enumerable.csv;

import java.lang.reflect.Type;
import java.util.concurrent.atomic.AtomicBoolean;

import org.imsi.queryEREngine.apache.calcite.DataContext;
import org.apache.calcite.linq4j.AbstractEnumerable;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.linq4j.QueryProvider;
import org.apache.calcite.linq4j.Queryable;
import org.apache.calcite.linq4j.tree.Expression;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelProtoDataType;
import org.imsi.queryEREngine.apache.calcite.schema.QueryableTable;
import org.imsi.queryEREngine.apache.calcite.schema.SchemaPlus;
import org.imsi.queryEREngine.apache.calcite.schema.Schemas;
import org.imsi.queryEREngine.apache.calcite.schema.TranslatableTable;
import org.imsi.queryEREngine.apache.calcite.util.Source;

/**
 * Table based on a CSV file.
 */
public class CsvTranslatableTable extends CsvTable
implements QueryableTable, TranslatableTable {



	/** Creates a CsvTable. */
	public CsvTranslatableTable(Source source, String name, RelProtoDataType protoRowType) {
		super(source, name, protoRowType);

	}

	@Override
	public String toString() {
		return "CsvTranslatableTable";
	}

	/** Returns an enumerable over a given projection of the fields.
	 *
	 * <p>Called from generated code. */
	public Enumerable<Object[]> project(final DataContext root,
			final Integer[] fields) {
		final AtomicBoolean cancelFlag = DataContext.Variable.CANCEL_FLAG.get(root);
		return new AbstractEnumerable<Object[]>() {
			@Override
			public Enumerator<Object[]> enumerator() {
				CsvEnumerator<Object[]> enumerator = new CsvEnumerator<Object[]>(source, cancelFlag, fieldTypes, fields, tableKey);
				return enumerator;
			}


		};

	}

	@Override
	public Expression getExpression(SchemaPlus schema, String tableName,
			Class clazz) {
		return Schemas.tableExpression(schema, getElementType(), tableName, clazz);
	}

	@Override
	public Type getElementType() {
		return Object[].class;
	}


	@Override
	public <T> Queryable<T> asQueryable(QueryProvider queryProvider,
			SchemaPlus schema, String tableName) {
		throw new UnsupportedOperationException();
	}

	@Override
	public RelNode toRel(
			RelOptTable.ToRelContext context,
			RelOptTable relOptTable) {
		// Request all fields.
		final int fieldCount = relOptTable.getRowType().getFieldCount();
		final Integer[] fields = CsvEnumerator.identityList(fieldCount);
		return new CsvTableScan(context.getCluster(), relOptTable, this, fields);
	}






}

// End CsvTranslatableTable.java
