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
package org.imsi.queryEREngine.apache.calcite.prepare;

import java.io.Reader;
import java.util.List;
import java.util.Properties;

import org.imsi.queryEREngine.apache.calcite.adapter.java.JavaTypeFactory;
import org.imsi.queryEREngine.apache.calcite.config.CalciteConnectionConfig;
import org.imsi.queryEREngine.apache.calcite.config.CalciteConnectionConfigImpl;
import org.imsi.queryEREngine.apache.calcite.config.CalciteConnectionProperty;
import org.imsi.queryEREngine.apache.calcite.config.CalciteSystemProperty;
import org.imsi.queryEREngine.apache.calcite.jdbc.CalciteSchema;
import org.imsi.queryEREngine.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.imsi.queryEREngine.apache.calcite.plan.Context;
import org.imsi.queryEREngine.apache.calcite.plan.ConventionTraitDef;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCostFactory;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptPlanner;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable.ViewExpander;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptUtil;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitDef;
import org.imsi.queryEREngine.apache.calcite.plan.RelTraitSet;
import org.imsi.queryEREngine.apache.calcite.plan.volcano.VolcanoPlanner;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollationTraitDef;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelRoot;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.CachingRelMetadataProvider;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeSystem;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexExecutor;
import org.imsi.queryEREngine.apache.calcite.runtime.Hook;
import org.imsi.queryEREngine.apache.calcite.schema.SchemaPlus;
import org.imsi.queryEREngine.apache.calcite.sql.SqlNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperatorTable;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParseException;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParser;
import org.imsi.queryEREngine.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlConformance;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlValidator;
import org.imsi.queryEREngine.apache.calcite.sql2rel.RelDecorrelator;
import org.imsi.queryEREngine.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.imsi.queryEREngine.apache.calcite.sql2rel.SqlToRelConverter;
import org.imsi.queryEREngine.apache.calcite.tools.FrameworkConfig;
import org.imsi.queryEREngine.apache.calcite.tools.Planner;
import org.imsi.queryEREngine.apache.calcite.tools.Program;
import org.imsi.queryEREngine.apache.calcite.tools.RelBuilder;
import org.imsi.queryEREngine.apache.calcite.tools.RelConversionException;
import org.imsi.queryEREngine.apache.calcite.tools.ValidationException;
import org.imsi.queryEREngine.apache.calcite.util.Pair;

import com.google.common.collect.ImmutableList;

/** Implementation of {@link org.imsi.queryEREngine.apache.calcite.tools.Planner}. */
public class PlannerImpl implements Planner, ViewExpander {
	private final SqlOperatorTable operatorTable;
	private final ImmutableList<Program> programs;
	private final RelOptCostFactory costFactory;
	private final Context context;
	private final CalciteConnectionConfig connectionConfig;

	/** Holds the trait definitions to be registered with planner. May be null. */
	private final ImmutableList<RelTraitDef> traitDefs;

	private final SqlParser.Config parserConfig;
	private final SqlToRelConverter.Config sqlToRelConverterConfig;
	private final SqlRexConvertletTable convertletTable;

	private State state;

	// set in STATE_1_RESET
	private boolean open;

	// set in STATE_2_READY
	private SchemaPlus defaultSchema;
	private JavaTypeFactory typeFactory;
	private RelOptPlanner planner;
	private RexExecutor executor;

	// set in STATE_4_VALIDATE
	private SqlValidator validator;
	private SqlNode validatedSqlNode;

	// set in STATE_5_CONVERT
	private RelRoot root;

	/** Creates a planner. Not a public API; call
	 * {@link org.imsi.queryEREngine.apache.calcite.tools.Frameworks#getPlanner} instead. */
	public PlannerImpl(FrameworkConfig config) {
		this.costFactory = config.getCostFactory();
		this.defaultSchema = config.getDefaultSchema();
		this.operatorTable = config.getOperatorTable();
		this.programs = config.getPrograms();
		this.parserConfig = config.getParserConfig();
		this.sqlToRelConverterConfig = config.getSqlToRelConverterConfig();
		this.state = State.STATE_0_CLOSED;
		this.traitDefs = config.getTraitDefs();
		this.convertletTable = config.getConvertletTable();
		this.executor = config.getExecutor();
		this.context = config.getContext();
		this.connectionConfig = connConfig();
		reset();
	}

	/** Gets a user defined config and appends default connection values */
	private CalciteConnectionConfig connConfig() {
		CalciteConnectionConfigImpl config =
				context.unwrap(CalciteConnectionConfigImpl.class);
		if (config == null) {
			config = new CalciteConnectionConfigImpl(new Properties());
		}
		if (!config.isSet(CalciteConnectionProperty.CASE_SENSITIVE)) {
			config = config.set(CalciteConnectionProperty.CASE_SENSITIVE,
					String.valueOf(parserConfig.caseSensitive()));
		}
		if (!config.isSet(CalciteConnectionProperty.CONFORMANCE)) {
			config = config.set(CalciteConnectionProperty.CONFORMANCE,
					String.valueOf(parserConfig.conformance()));
		}
		return config;
	}

	/** Makes sure that the state is at least the given state. */
	private void ensure(State state) {
		if (state == this.state) {
			return;
		}
		if (state.ordinal() < this.state.ordinal()) {
			throw new IllegalArgumentException("cannot move to " + state + " from "
					+ this.state);
		}
		state.from(this);
	}

	@Override
	public RelTraitSet getEmptyTraitSet() {
		return planner.emptyTraitSet();
	}

	@Override
	public void close() {
		open = false;
		typeFactory = null;
		state = State.STATE_0_CLOSED;
	}

	@Override
	public void reset() {
		ensure(State.STATE_0_CLOSED);
		open = true;
		state = State.STATE_1_RESET;
	}

	private void ready() {
		switch (state) {
		case STATE_0_CLOSED:
			reset();
		}
		ensure(State.STATE_1_RESET);

		RelDataTypeSystem typeSystem =
				connectionConfig.typeSystem(RelDataTypeSystem.class,
						RelDataTypeSystem.DEFAULT);
		typeFactory = new JavaTypeFactoryImpl(typeSystem);
		planner = new VolcanoPlanner(costFactory, context);
		RelOptUtil.registerDefaultRules(planner,
				connectionConfig.materializationsEnabled(),
				Hook.ENABLE_BINDABLE.get(false));
		planner.setExecutor(executor);

		state = State.STATE_2_READY;

		// If user specify own traitDef, instead of default default trait,
		// register the trait def specified in traitDefs.
		if (this.traitDefs == null) {
			planner.addRelTraitDef(ConventionTraitDef.INSTANCE);
			if (CalciteSystemProperty.ENABLE_COLLATION_TRAIT.value()) {
				planner.addRelTraitDef(RelCollationTraitDef.INSTANCE);
			}
		} else {
			for (RelTraitDef def : this.traitDefs) {
				planner.addRelTraitDef(def);
			}
		}
	}

	@Override
	public SqlNode parse(final Reader reader) throws SqlParseException {
		switch (state) {
		case STATE_0_CLOSED:
		case STATE_1_RESET:
			ready();
		}
		ensure(State.STATE_2_READY);
		SqlParser parser = SqlParser.create(reader, parserConfig);
		SqlNode sqlNode = parser.parseStmt();
		state = State.STATE_3_PARSED;
		return sqlNode;
	}

	@Override
	public SqlNode validate(SqlNode sqlNode) throws ValidationException {
		ensure(State.STATE_3_PARSED);
		this.validator = createSqlValidator(createCatalogReader());
		try {
			validatedSqlNode = validator.validate(sqlNode);
		} catch (RuntimeException e) {
			throw new ValidationException(e);
		}
		state = State.STATE_4_VALIDATED;
		return validatedSqlNode;
	}

	private SqlConformance conformance() {
		return connectionConfig.conformance();
	}

	@Override
	public Pair<SqlNode, RelDataType> validateAndGetType(SqlNode sqlNode)
			throws ValidationException {
		final SqlNode validatedNode = this.validate(sqlNode);
		final RelDataType type =
				this.validator.getValidatedNodeType(validatedNode);
		return Pair.of(validatedNode, type);
	}

	@Override
	@SuppressWarnings("deprecation")
	public final RelNode convert(SqlNode sql) throws RelConversionException {
		return rel(sql).rel;
	}

	@Override
	public RelRoot rel(SqlNode sql) throws RelConversionException {
		ensure(State.STATE_4_VALIDATED);
		assert validatedSqlNode != null;

		final RexBuilder rexBuilder = createRexBuilder();
		final RelOptCluster cluster = RelOptCluster.create(planner, rexBuilder);
		final SqlToRelConverter.Config config = SqlToRelConverter.configBuilder()
				.withConfig(sqlToRelConverterConfig)
				.withTrimUnusedFields(false)
				.build();
		final SqlToRelConverter sqlToRelConverter =
				new SqlToRelConverter(this, validator,
						createCatalogReader(), cluster, convertletTable, config);
		root =
				sqlToRelConverter.convertQuery(validatedSqlNode, false, true);
		root = root.withRel(sqlToRelConverter.flattenTypes(root.rel, true));

		final RelBuilder relBuilder =
				config.getRelBuilderFactory().create(cluster, null);
		root = root.withRel(
				RelDecorrelator.decorrelateQuery(root.rel, relBuilder));
		return root;
	}

	/** @deprecated Now {@link PlannerImpl} implements {@link ViewExpander}
	 * directly. */
	@Deprecated
	public class ViewExpanderImpl implements ViewExpander {
		ViewExpanderImpl() {
		}

		@Override
		public RelRoot expandView(RelDataType rowType, String queryString,
				List<String> schemaPath, List<String> viewPath) {
			return PlannerImpl.this.expandView(rowType, queryString, schemaPath,
					viewPath);
		}
	}

	@Override public RelRoot expandView(RelDataType rowType, String queryString,
			List<String> schemaPath, List<String> viewPath) {
		if (planner == null) {
			ready();
		}
		SqlParser parser = SqlParser.create(queryString, parserConfig);
		SqlNode sqlNode;
		try {
			sqlNode = parser.parseQuery();
		} catch (SqlParseException e) {
			throw new RuntimeException("parse failed", e);
		}

		final CalciteCatalogReader catalogReader =
				createCatalogReader().withSchemaPath(schemaPath);
		final SqlValidator validator = createSqlValidator(catalogReader);

		final RexBuilder rexBuilder = createRexBuilder();
		final RelOptCluster cluster = RelOptCluster.create(planner, rexBuilder);
		final SqlToRelConverter.Config config = SqlToRelConverter
				.configBuilder()
				.withConfig(sqlToRelConverterConfig)
				.withTrimUnusedFields(false)
				.build();
		final SqlToRelConverter sqlToRelConverter =
				new SqlToRelConverter(this, validator,
						catalogReader, cluster, convertletTable, config);

		final RelRoot root =
				sqlToRelConverter.convertQuery(sqlNode, true, false);

		final RelRoot root2 =
				root.withRel(sqlToRelConverter.flattenTypes(root.rel, true));
		final RelBuilder relBuilder =
				config.getRelBuilderFactory().create(cluster, null);
		return root2.withRel(
				RelDecorrelator.decorrelateQuery(root.rel, relBuilder));
	}

	// CalciteCatalogReader is stateless; no need to store one
	private CalciteCatalogReader createCatalogReader() {
		final SchemaPlus rootSchema = rootSchema(defaultSchema);

		return new CalciteCatalogReader(
				CalciteSchema.from(rootSchema),
				CalciteSchema.from(defaultSchema).path(null),
				typeFactory, connectionConfig);
	}

	private SqlValidator createSqlValidator(CalciteCatalogReader catalogReader) {
		final SqlConformance conformance = conformance();
		final SqlOperatorTable opTab =
				ChainedSqlOperatorTable.of(operatorTable, catalogReader);
		final SqlValidator validator =
				new CalciteSqlValidator(opTab, catalogReader, typeFactory, conformance);
		validator.setIdentifierExpansion(true);
		return validator;
	}

	private static SchemaPlus rootSchema(SchemaPlus schema) {
		for (;;) {
			if (schema.getParentSchema() == null) {
				return schema;
			}
			schema = schema.getParentSchema();
		}
	}

	// RexBuilder is stateless; no need to store one
	private RexBuilder createRexBuilder() {
		return new RexBuilder(typeFactory);
	}

	@Override
	public JavaTypeFactory getTypeFactory() {
		return typeFactory;
	}

	@Override
	public RelNode transform(int ruleSetIndex, RelTraitSet requiredOutputTraits,
			RelNode rel) throws RelConversionException {
		ensure(State.STATE_5_CONVERTED);
		rel.getCluster().setMetadataProvider(
				new CachingRelMetadataProvider(
						rel.getCluster().getMetadataProvider(),
						rel.getCluster().getPlanner()));
		Program program = programs.get(ruleSetIndex);
		return program.run(planner, rel, requiredOutputTraits, ImmutableList.of(),
				ImmutableList.of());
	}

	/** Stage of a statement in the query-preparation lifecycle. */
	private enum State {
		STATE_0_CLOSED {
			@Override void from(PlannerImpl planner) {
				planner.close();
			}
		},
		STATE_1_RESET {
			@Override void from(PlannerImpl planner) {
				planner.ensure(STATE_0_CLOSED);
				planner.reset();
			}
		},
		STATE_2_READY {
			@Override void from(PlannerImpl planner) {
				STATE_1_RESET.from(planner);
				planner.ready();
			}
		},
		STATE_3_PARSED,
		STATE_4_VALIDATED,
		STATE_5_CONVERTED;

		/** Moves planner's state to this state. This must be a higher state. */
		void from(PlannerImpl planner) {
			throw new IllegalArgumentException("cannot move from " + planner.state
					+ " to " + this);
		}
	}
}
