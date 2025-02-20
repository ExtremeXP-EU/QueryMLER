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
package org.imsi.queryEREngine.apache.calcite.adapter.clone;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;

import org.imsi.queryEREngine.apache.calcite.adapter.java.AbstractQueryableTable;
import org.apache.calcite.linq4j.AbstractQueryable;
import org.apache.calcite.linq4j.Enumerator;
import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.linq4j.QueryProvider;
import org.apache.calcite.linq4j.Queryable;
import org.apache.calcite.linq4j.tree.Expression;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelProtoDataType;
import org.imsi.queryEREngine.apache.calcite.schema.SchemaPlus;
import org.imsi.queryEREngine.apache.calcite.schema.Statistic;
import org.imsi.queryEREngine.apache.calcite.schema.Statistics;

import com.google.common.collect.ImmutableList;

/**
 * Implementation of table that reads rows from a read-only list and returns
 * an enumerator of rows. Each row is object (if there is just one column) or
 * an object array (if there are multiple columns).
 */
class ListTable extends AbstractQueryableTable {
	private final RelProtoDataType protoRowType;
	private final Expression expression;
	private final List list;

	/** Creates a ListTable. */
	ListTable(
			Type elementType,
			RelProtoDataType protoRowType,
			Expression expression,
			List list) {
		super(elementType);
		this.protoRowType = protoRowType;
		this.expression = expression;
		this.list = list;
	}

	@Override
	public RelDataType getRowType(RelDataTypeFactory typeFactory) {
		return protoRowType.apply(typeFactory);
	}

	@Override
	public Statistic getStatistic() {
		return Statistics.of(list.size(), ImmutableList.of());
	}

	@Override
	public <T> Queryable<T> asQueryable(final QueryProvider queryProvider,
			SchemaPlus schema, String tableName) {
		return new AbstractQueryable<T>() {
			@Override
			public Type getElementType() {
				return elementType;
			}

			@Override
			public Expression getExpression() {
				return expression;
			}

			@Override
			public QueryProvider getProvider() {
				return queryProvider;
			}

			@Override
			public Iterator<T> iterator() {
				//noinspection unchecked
				return list.iterator();
			}

			@Override
			public Enumerator<T> enumerator() {
				//noinspection unchecked
				return Linq4j.enumerator(list);
			}
		};
	}
}
