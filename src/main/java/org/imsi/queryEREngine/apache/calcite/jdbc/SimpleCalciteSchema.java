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
package org.imsi.queryEREngine.apache.calcite.jdbc;

import java.util.Collection;
import java.util.List;

import org.imsi.queryEREngine.apache.calcite.rel.type.RelProtoDataType;
import org.imsi.queryEREngine.apache.calcite.schema.Function;
import org.imsi.queryEREngine.apache.calcite.schema.Schema;
import org.imsi.queryEREngine.apache.calcite.schema.SchemaVersion;
import org.imsi.queryEREngine.apache.calcite.schema.Table;
import org.imsi.queryEREngine.apache.calcite.schema.TableMacro;
import org.imsi.queryEREngine.apache.calcite.util.NameMap;
import org.imsi.queryEREngine.apache.calcite.util.NameMultimap;
import org.imsi.queryEREngine.apache.calcite.util.NameSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;

/**
 * A concrete implementation of {@link org.imsi.queryEREngine.apache.calcite.jdbc.CalciteSchema}
 * that maintains minimal state.
 */
class SimpleCalciteSchema extends CalciteSchema {
	/** Creates a SimpleCalciteSchema.
	 *
	 * <p>Use {@link CalciteSchema#createRootSchema(boolean)}
	 * or {@link #add(String, Schema)}. */
	SimpleCalciteSchema(CalciteSchema parent, Schema schema, String name) {
		this(parent, schema, name, null, null, null, null, null, null, null, null);
	}

	private SimpleCalciteSchema(CalciteSchema parent, Schema schema,
			String name, NameMap<CalciteSchema> subSchemaMap,
			NameMap<TableEntry> tableMap, NameMap<LatticeEntry> latticeMap, NameMap<TypeEntry> typeMap,
			NameMultimap<FunctionEntry> functionMap, NameSet functionNames,
			NameMap<FunctionEntry> nullaryFunctionMap,
			List<? extends List<String>> path) {
		super(parent, schema, name, subSchemaMap, tableMap, latticeMap, typeMap,
				functionMap, functionNames, nullaryFunctionMap, path);
	}

	@Override
	public void setCache(boolean cache) {
		throw new UnsupportedOperationException();
	}

	@Override
	public CalciteSchema add(String name, Schema schema) {
		final CalciteSchema calciteSchema =
				new SimpleCalciteSchema(this, schema, name);
		subSchemaMap.put(name, calciteSchema);
		return calciteSchema;
	}

	@Override
	protected CalciteSchema getImplicitSubSchema(String schemaName,
			boolean caseSensitive) {
		// Check implicit schemas.
		Schema s = schema.getSubSchema(schemaName);
		if (s != null) {
			return new SimpleCalciteSchema(this, s, schemaName);
		}
		return null;
	}

	@Override
	protected TableEntry getImplicitTable(String tableName,
			boolean caseSensitive) {
		// Check implicit tables.
		Table table = schema.getTable(tableName);
		if (table != null) {
			return tableEntry(tableName, table);
		}
		return null;
	}

	@Override
	protected TypeEntry getImplicitType(String name, boolean caseSensitive) {
		// Check implicit types.
		RelProtoDataType type = schema.getType(name);
		if (type != null) {
			return typeEntry(name, type);
		}
		return null;
	}

	@Override
	protected void addImplicitSubSchemaToBuilder(
			ImmutableSortedMap.Builder<String, CalciteSchema> builder) {
		ImmutableSortedMap<String, CalciteSchema> explicitSubSchemas = builder.build();
		for (String schemaName : schema.getSubSchemaNames()) {
			if (explicitSubSchemas.containsKey(schemaName)) {
				// explicit subschema wins.
				continue;
			}
			Schema s = schema.getSubSchema(schemaName);
			if (s != null) {
				CalciteSchema calciteSchema = new SimpleCalciteSchema(this, s, schemaName);
				builder.put(schemaName, calciteSchema);
			}
		}
	}

	@Override
	protected void addImplicitTableToBuilder(ImmutableSortedSet.Builder<String> builder) {
		builder.addAll(schema.getTableNames());
	}

	@Override
	protected void addImplicitFunctionsToBuilder(
			ImmutableList.Builder<Function> builder,
			String name, boolean caseSensitive) {
		Collection<Function> functions = schema.getFunctions(name);
		if (functions != null) {
			builder.addAll(functions);
		}
	}

	@Override
	protected void addImplicitFuncNamesToBuilder(ImmutableSortedSet.Builder<String> builder) {
		builder.addAll(schema.getFunctionNames());
	}

	@Override protected void addImplicitTypeNamesToBuilder(
			ImmutableSortedSet.Builder<String> builder) {
		builder.addAll(schema.getTypeNames());
	}

	@Override
	protected void addImplicitTablesBasedOnNullaryFunctionsToBuilder(
			ImmutableSortedMap.Builder<String, Table> builder) {
		ImmutableSortedMap<String, Table> explicitTables = builder.build();

		for (String s : schema.getFunctionNames()) {
			// explicit table wins.
			if (explicitTables.containsKey(s)) {
				continue;
			}
			for (Function function : schema.getFunctions(s)) {
				if (function instanceof TableMacro
						&& function.getParameters().isEmpty()) {
					final Table table = ((TableMacro) function).apply(ImmutableList.of());
					builder.put(s, table);
				}
			}
		}
	}

	@Override
	protected TableEntry getImplicitTableBasedOnNullaryFunction(String tableName,
			boolean caseSensitive) {
		Collection<Function> functions = schema.getFunctions(tableName);
		if (functions != null) {
			for (Function function : functions) {
				if (function instanceof TableMacro
						&& function.getParameters().isEmpty()) {
					final Table table = ((TableMacro) function).apply(ImmutableList.of());
					return tableEntry(tableName, table);
				}
			}
		}
		return null;
	}

	@Override
	protected CalciteSchema snapshot(CalciteSchema parent, SchemaVersion version) {
		CalciteSchema snapshot = new SimpleCalciteSchema(parent,
				schema.snapshot(version), name, null, tableMap, latticeMap, typeMap,
				functionMap, functionNames, nullaryFunctionMap, getPath());
		for (CalciteSchema subSchema : subSchemaMap.map().values()) {
			CalciteSchema subSchemaSnapshot = subSchema.snapshot(snapshot, version);
			snapshot.subSchemaMap.put(subSchema.name, subSchemaSnapshot);
		}
		return snapshot;
	}

	@Override
	protected boolean isCacheEnabled() {
		return false;
	}

}
