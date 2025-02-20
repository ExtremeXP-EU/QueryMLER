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
package org.imsi.queryEREngine.apache.calcite.rel.externalize;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.calcite.avatica.AvaticaUtils;
import org.apache.calcite.avatica.util.TimeUnit;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollationImpl;
import org.imsi.queryEREngine.apache.calcite.rel.RelCollations;
import org.imsi.queryEREngine.apache.calcite.rel.RelDistribution;
import org.imsi.queryEREngine.apache.calcite.rel.RelDistributions;
import org.imsi.queryEREngine.apache.calcite.rel.RelFieldCollation;
import org.imsi.queryEREngine.apache.calcite.rel.RelFieldCollation.Direction;
import org.imsi.queryEREngine.apache.calcite.rel.RelFieldCollation.NullDirection;
import org.imsi.queryEREngine.apache.calcite.rel.RelInput;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.AggregateCall;
import org.imsi.queryEREngine.apache.calcite.rel.core.CorrelationId;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeField;
import org.imsi.queryEREngine.apache.calcite.rex.RexBuilder;
import org.imsi.queryEREngine.apache.calcite.rex.RexCall;
import org.imsi.queryEREngine.apache.calcite.rex.RexCorrelVariable;
import org.imsi.queryEREngine.apache.calcite.rex.RexFieldAccess;
import org.imsi.queryEREngine.apache.calcite.rex.RexFieldCollation;
import org.imsi.queryEREngine.apache.calcite.rex.RexLiteral;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.rex.RexOver;
import org.imsi.queryEREngine.apache.calcite.rex.RexSlot;
import org.imsi.queryEREngine.apache.calcite.rex.RexWindow;
import org.imsi.queryEREngine.apache.calcite.rex.RexWindowBound;
import org.imsi.queryEREngine.apache.calcite.sql.SqlAggFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlFunction;
import org.imsi.queryEREngine.apache.calcite.sql.SqlIdentifier;
import org.imsi.queryEREngine.apache.calcite.sql.SqlIntervalQualifier;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperator;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSyntax;
import org.imsi.queryEREngine.apache.calcite.sql.SqlWindow;
import org.imsi.queryEREngine.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.sql.validate.SqlNameMatchers;
import org.imsi.queryEREngine.apache.calcite.util.ImmutableBitSet;
import org.imsi.queryEREngine.apache.calcite.util.JsonBuilder;
import org.imsi.queryEREngine.apache.calcite.util.Util;

import com.google.common.collect.ImmutableList;

/**
 * Utilities for converting {@link org.imsi.queryEREngine.apache.calcite.rel.RelNode}
 * into JSON format.
 */
public class RelJson {
	private final Map<String, Constructor> constructorMap = new HashMap<>();
	private final JsonBuilder jsonBuilder;

	public static final List<String> PACKAGES =
			ImmutableList.of(
					"org.imsi.queryEREngine.apache.calcite.rel.",
					"org.imsi.queryEREngine.apache.calcite.rel.core.",
					"org.imsi.queryEREngine.apache.calcite.rel.logical.",
					"org.imsi.queryEREngine.apache.calcite.adapter.jdbc.",
					"org.imsi.queryEREngine.apache.calcite.adapter.jdbc.JdbcRules$");

	public RelJson(JsonBuilder jsonBuilder) {
		this.jsonBuilder = jsonBuilder;
	}

	public RelNode create(Map<String, Object> map) {
		String type = (String) map.get("type");
		Constructor constructor = getConstructor(type);
		try {
			return (RelNode) constructor.newInstance(map);
		} catch (InstantiationException | ClassCastException | InvocationTargetException
				| IllegalAccessException e) {
			throw new RuntimeException(
					"while invoking constructor for type '" + type + "'", e);
		}
	}

	public Constructor getConstructor(String type) {
		Constructor constructor = constructorMap.get(type);
		if (constructor == null) {
			Class clazz = typeNameToClass(type);
			try {
				//noinspection unchecked
				constructor = clazz.getConstructor(RelInput.class);
			} catch (NoSuchMethodException e) {
				throw new RuntimeException("class does not have required constructor, "
						+ clazz + "(RelInput)");
			}
			constructorMap.put(type, constructor);
		}
		return constructor;
	}

	/**
	 * Converts a type name to a class. E.g. {@code getClass("LogicalProject")}
	 * returns {@link org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalProject}.class.
	 */
	public Class typeNameToClass(String type) {
		if (!type.contains(".")) {
			for (String package_ : PACKAGES) {
				try {
					return Class.forName(package_ + type);
				} catch (ClassNotFoundException e) {
					// ignore
				}
			}
		}
		try {
			return Class.forName(type);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException("unknown type " + type);
		}
	}

	/**
	 * Inverse of {@link #typeNameToClass}.
	 */
	public String classToTypeName(Class<? extends RelNode> class_) {
		final String canonicalName = class_.getName();
		for (String package_ : PACKAGES) {
			if (canonicalName.startsWith(package_)) {
				String remaining = canonicalName.substring(package_.length());
				if (remaining.indexOf('.') < 0 && remaining.indexOf('$') < 0) {
					return remaining;
				}
			}
		}
		return canonicalName;
	}

	public Object toJson(RelCollationImpl node) {
		final List<Object> list = new ArrayList<>();
		for (RelFieldCollation fieldCollation : node.getFieldCollations()) {
			final Map<String, Object> map = jsonBuilder.map();
			map.put("field", fieldCollation.getFieldIndex());
			map.put("direction", fieldCollation.getDirection().name());
			map.put("nulls", fieldCollation.nullDirection.name());
			list.add(map);
		}
		return list;
	}

	public RelCollation toCollation(
			List<Map<String, Object>> jsonFieldCollations) {
		final List<RelFieldCollation> fieldCollations = new ArrayList<>();
		for (Map<String, Object> map : jsonFieldCollations) {
			fieldCollations.add(toFieldCollation(map));
		}
		return RelCollations.of(fieldCollations);
	}

	public RelFieldCollation toFieldCollation(Map<String, Object> map) {
		final Integer field = (Integer) map.get("field");
		final RelFieldCollation.Direction direction =
				Util.enumVal(RelFieldCollation.Direction.class,
						(String) map.get("direction"));
		final RelFieldCollation.NullDirection nullDirection =
				Util.enumVal(RelFieldCollation.NullDirection.class,
						(String) map.get("nulls"));
		return new RelFieldCollation(field, direction, nullDirection);
	}

	public RelDistribution toDistribution(Object o) {
		return RelDistributions.ANY; // TODO:
	}

	public RelDataType toType(RelDataTypeFactory typeFactory, Object o) {
		if (o instanceof List) {
			@SuppressWarnings("unchecked")
			final List<Map<String, Object>> jsonList = (List<Map<String, Object>>) o;
			final RelDataTypeFactory.Builder builder = typeFactory.builder();
			for (Map<String, Object> jsonMap : jsonList) {
				builder.add((String) jsonMap.get("name"), toType(typeFactory, jsonMap));
			}
			return builder.build();
		} else if (o instanceof Map) {
			@SuppressWarnings("unchecked")
			final Map<String, Object> map = (Map<String, Object>) o;
			final Object fields = map.get("fields");
			if (fields != null) {
				// Nested struct
				return toType(typeFactory, fields);
			} else {
				final SqlTypeName sqlTypeName =
						Util.enumVal(SqlTypeName.class, (String) map.get("type"));
				final Integer precision = (Integer) map.get("precision");
				final Integer scale = (Integer) map.get("scale");
				if (SqlTypeName.INTERVAL_TYPES.contains(sqlTypeName)) {
					TimeUnit startUnit = sqlTypeName.getStartUnit();
					TimeUnit endUnit = sqlTypeName.getEndUnit();
					return typeFactory.createSqlIntervalType(
							new SqlIntervalQualifier(startUnit, endUnit, SqlParserPos.ZERO));
				}
				final RelDataType type;
				if (precision == null) {
					type = typeFactory.createSqlType(sqlTypeName);
				} else if (scale == null) {
					type = typeFactory.createSqlType(sqlTypeName, precision);
				} else {
					type = typeFactory.createSqlType(sqlTypeName, precision, scale);
				}
				final boolean nullable = (Boolean) map.get("nullable");
				return typeFactory.createTypeWithNullability(type, nullable);
			}
		} else {
			final SqlTypeName sqlTypeName =
					Util.enumVal(SqlTypeName.class, (String) o);
			return typeFactory.createSqlType(sqlTypeName);
		}
	}

	public Object toJson(AggregateCall node) {
		final Map<String, Object> map = jsonBuilder.map();
		map.put("agg", toJson(node.getAggregation()));
		map.put("type", toJson(node.getType()));
		map.put("distinct", node.isDistinct());
		map.put("operands", node.getArgList());
		map.put("name", node.getName());
		return map;
	}

	public Object toJson(Object value) {
		if (value == null
				|| value instanceof Number
				|| value instanceof String
				|| value instanceof Boolean) {
			return value;
		} else if (value instanceof RexNode) {
			return toJson((RexNode) value);
		} else if (value instanceof RexWindow) {
			return toJson((RexWindow) value);
		} else if (value instanceof RexFieldCollation) {
			return toJson((RexFieldCollation) value);
		} else if (value instanceof RexWindowBound) {
			return toJson((RexWindowBound) value);
		} else if (value instanceof CorrelationId) {
			return toJson((CorrelationId) value);
		} else if (value instanceof List) {
			final List<Object> list = jsonBuilder.list();
			for (Object o : (List) value) {
				list.add(toJson(o));
			}
			return list;
		} else if (value instanceof ImmutableBitSet) {
			final List<Object> list = jsonBuilder.list();
			for (Integer integer : (ImmutableBitSet) value) {
				list.add(toJson(integer));
			}
			return list;
		} else if (value instanceof AggregateCall) {
			return toJson((AggregateCall) value);
		} else if (value instanceof RelCollationImpl) {
			return toJson((RelCollationImpl) value);
		} else if (value instanceof RelDataType) {
			return toJson((RelDataType) value);
		} else if (value instanceof RelDataTypeField) {
			return toJson((RelDataTypeField) value);
		} else {
			throw new UnsupportedOperationException("type not serializable: "
					+ value + " (type " + value.getClass().getCanonicalName() + ")");
		}
	}

	private Object toJson(RelDataType node) {
		if (node.isStruct()) {
			final List<Object> list = jsonBuilder.list();
			for (RelDataTypeField field : node.getFieldList()) {
				list.add(toJson(field));
			}
			return list;
		} else {
			final Map<String, Object> map = jsonBuilder.map();
			map.put("type", node.getSqlTypeName().name());
			map.put("nullable", node.isNullable());
			if (node.getSqlTypeName().allowsPrec()) {
				map.put("precision", node.getPrecision());
			}
			if (node.getSqlTypeName().allowsScale()) {
				map.put("scale", node.getScale());
			}
			return map;
		}
	}

	private Object toJson(RelDataTypeField node) {
		final Map<String, Object> map;
		if (node.getType().isStruct()) {
			map = jsonBuilder.map();
			map.put("fields", toJson(node.getType()));
		} else {
			map = (Map<String, Object>) toJson(node.getType());
		}
		map.put("name", node.getName());
		return map;
	}

	private Object toJson(CorrelationId node) {
		return node.getId();
	}

	private Object toJson(RexNode node) {
		final Map<String, Object> map;
		switch (node.getKind()) {
		case FIELD_ACCESS:
			map = jsonBuilder.map();
			final RexFieldAccess fieldAccess = (RexFieldAccess) node;
			map.put("field", fieldAccess.getField().getName());
			map.put("expr", toJson(fieldAccess.getReferenceExpr()));
			return map;
		case LITERAL:
			final RexLiteral literal = (RexLiteral) node;
			final Object value = literal.getValue3();
			map = jsonBuilder.map();
			map.put("literal", RelEnumTypes.fromEnum(value));
			map.put("type", toJson(node.getType()));
			return map;
		case INPUT_REF:
			map = jsonBuilder.map();
			map.put("input", ((RexSlot) node).getIndex());
			map.put("name", ((RexSlot) node).getName());
			return map;
		case LOCAL_REF:
			map = jsonBuilder.map();
			map.put("input", ((RexSlot) node).getIndex());
			map.put("name", ((RexSlot) node).getName());
			map.put("type", toJson(node.getType()));
			return map;
		case CORREL_VARIABLE:
			map = jsonBuilder.map();
			map.put("correl", ((RexCorrelVariable) node).getName());
			map.put("type", toJson(node.getType()));
			return map;
		default:
			if (node instanceof RexCall) {
				final RexCall call = (RexCall) node;
				map = jsonBuilder.map();
				map.put("op", toJson(call.getOperator()));
				final List<Object> list = jsonBuilder.list();
				for (RexNode operand : call.getOperands()) {
					list.add(toJson(operand));
				}
				map.put("operands", list);
				switch (node.getKind()) {
				case CAST:
					map.put("type", toJson(node.getType()));
				}
				if (call.getOperator() instanceof SqlFunction) {
					if (((SqlFunction) call.getOperator()).getFunctionType().isUserDefined()) {
						SqlOperator op = call.getOperator();
						map.put("class", op.getClass().getName());
						map.put("type", toJson(node.getType()));
						map.put("deterministic", op.isDeterministic());
						map.put("dynamic", op.isDynamicFunction());
					}
				}
				if (call instanceof RexOver) {
					RexOver over = (RexOver) call;
					map.put("distinct", over.isDistinct());
					map.put("type", toJson(node.getType()));
					map.put("window", toJson(over.getWindow()));
				}
				return map;
			}
			throw new UnsupportedOperationException("unknown rex " + node);
		}
	}

	private Object toJson(RexWindow window) {
		final Map<String, Object> map = jsonBuilder.map();
		if (window.partitionKeys.size() > 0) {
			map.put("partition", toJson(window.partitionKeys));
		}
		if (window.orderKeys.size() > 0) {
			map.put("order", toJson(window.orderKeys));
		}
		if (window.getLowerBound() == null) {
			// No ROWS or RANGE clause
		} else if (window.getUpperBound() == null) {
			if (window.isRows()) {
				map.put("rows-lower", toJson(window.getLowerBound()));
			} else {
				map.put("range-lower", toJson(window.getLowerBound()));
			}
		} else {
			if (window.isRows()) {
				map.put("rows-lower", toJson(window.getLowerBound()));
				map.put("rows-upper", toJson(window.getUpperBound()));
			} else {
				map.put("range-lower", toJson(window.getLowerBound()));
				map.put("range-upper", toJson(window.getUpperBound()));
			}
		}
		return map;
	}

	private Object toJson(RexFieldCollation collation) {
		final Map<String, Object> map = jsonBuilder.map();
		map.put("expr", toJson(collation.left));
		map.put("direction", collation.getDirection().name());
		map.put("null-direction", collation.getNullDirection().name());
		return map;
	}

	private Object toJson(RexWindowBound windowBound) {
		final Map<String, Object> map = jsonBuilder.map();
		if (windowBound.isCurrentRow()) {
			map.put("type", "CURRENT_ROW");
		} else if (windowBound.isUnbounded()) {
			map.put("type", windowBound.isPreceding() ? "UNBOUNDED_PRECEDING" : "UNBOUNDED_FOLLOWING");
		} else {
			map.put("type", windowBound.isPreceding() ? "PRECEDING" : "FOLLOWING");
			map.put("offset", toJson(windowBound.getOffset()));
		}
		return map;
	}

	RexNode toRex(RelInput relInput, Object o) {
		final RelOptCluster cluster = relInput.getCluster();
		final RexBuilder rexBuilder = cluster.getRexBuilder();
		if (o == null) {
			return null;
		} else if (o instanceof Map) {
			Map map = (Map) o;
			final Map<String, Object> opMap = (Map) map.get("op");
			final RelDataTypeFactory typeFactory = cluster.getTypeFactory();
			if (opMap != null) {
				if (map.containsKey("class")) {
					opMap.put("class", map.get("class"));
				}
				final List operands = (List) map.get("operands");
				final List<RexNode> rexOperands = toRexList(relInput, operands);
				final Object jsonType = map.get("type");
				final Map window = (Map) map.get("window");
				if (window != null) {
					final SqlAggFunction operator = toAggregation(opMap);
					final RelDataType type = toType(typeFactory, jsonType);
					List<RexNode> partitionKeys = new ArrayList<>();
					if (window.containsKey("partition")) {
						partitionKeys = toRexList(relInput, (List) window.get("partition"));
					}
					List<RexFieldCollation> orderKeys = new ArrayList<>();
					if (window.containsKey("order")) {
						orderKeys = toRexFieldCollationList(relInput, (List) window.get("order"));
					}
					final RexWindowBound lowerBound;
					final RexWindowBound upperBound;
					final boolean physical;
					if (window.get("rows-lower") != null) {
						lowerBound = toRexWindowBound(relInput, (Map) window.get("rows-lower"));
						upperBound = toRexWindowBound(relInput, (Map) window.get("rows-upper"));
						physical = true;
					} else if (window.get("range-lower") != null) {
						lowerBound = toRexWindowBound(relInput, (Map) window.get("range-lower"));
						upperBound = toRexWindowBound(relInput, (Map) window.get("range-upper"));
						physical = false;
					} else {
						// No ROWS or RANGE clause
						lowerBound = null;
						upperBound = null;
						physical = false;
					}
					final boolean distinct = (Boolean) map.get("distinct");
					return rexBuilder.makeOver(type, operator, rexOperands, partitionKeys,
							ImmutableList.copyOf(orderKeys), lowerBound, upperBound, physical,
							true, false, distinct, false);
				} else {
					final SqlOperator operator = toOp(opMap);
					final RelDataType type;
					if (jsonType != null) {
						type = toType(typeFactory, jsonType);
					} else {
						type = rexBuilder.deriveReturnType(operator, rexOperands);
					}
					return rexBuilder.makeCall(type, operator, rexOperands);
				}
			}
			final Integer input = (Integer) map.get("input");
			if (input != null) {
				// Check if it is a local ref.
				if (map.containsKey("type")) {
					final RelDataType type = toType(typeFactory, map.get("type"));
					return rexBuilder.makeLocalRef(type, input);
				}

				List<RelNode> inputNodes = relInput.getInputs();
				int i = input;
				for (RelNode inputNode : inputNodes) {
					final RelDataType rowType = inputNode.getRowType();
					if (i < rowType.getFieldCount()) {
						final RelDataTypeField field = rowType.getFieldList().get(i);
						return rexBuilder.makeInputRef(field.getType(), input);
					}
					i -= rowType.getFieldCount();
				}
				throw new RuntimeException("input field " + input + " is out of range");
			}
			final String field = (String) map.get("field");
			if (field != null) {
				final Object jsonExpr = map.get("expr");
				final RexNode expr = toRex(relInput, jsonExpr);
				return rexBuilder.makeFieldAccess(expr, field, true);
			}
			final String correl = (String) map.get("correl");
			if (correl != null) {
				final Object jsonType = map.get("type");
				RelDataType type = toType(typeFactory, jsonType);
				return rexBuilder.makeCorrel(type, new CorrelationId(correl));
			}
			if (map.containsKey("literal")) {
				Object literal = map.get("literal");
				final RelDataType type = toType(typeFactory, map.get("type"));
				if (literal == null) {
					return rexBuilder.makeNullLiteral(type);
				}
				if (type == null) {
					// In previous versions, type was not specified for all literals.
					// To keep backwards compatibility, if type is not specified
					// we just interpret the literal
					return toRex(relInput, literal);
				}
				if (type.getSqlTypeName() == SqlTypeName.SYMBOL) {
					literal = RelEnumTypes.toEnum((String) literal);
				}
				return rexBuilder.makeLiteral(literal, type, false);
			}
			throw new UnsupportedOperationException("cannot convert to rex " + o);
		} else if (o instanceof Boolean) {
			return rexBuilder.makeLiteral((Boolean) o);
		} else if (o instanceof String) {
			return rexBuilder.makeLiteral((String) o);
		} else if (o instanceof Number) {
			final Number number = (Number) o;
			if (number instanceof Double || number instanceof Float) {
				return rexBuilder.makeApproxLiteral(
						BigDecimal.valueOf(number.doubleValue()));
			} else {
				return rexBuilder.makeExactLiteral(
						BigDecimal.valueOf(number.longValue()));
			}
		} else {
			throw new UnsupportedOperationException("cannot convert to rex " + o);
		}
	}

	private List<RexFieldCollation> toRexFieldCollationList(
			RelInput relInput, List<Map<String, Object>> order) {
		if (order == null) {
			return null;
		}

		List<RexFieldCollation> list = new ArrayList<>();
		for (Map<String, Object> o : order) {
			RexNode expr = toRex(relInput, o.get("expr"));
			Set<SqlKind> directions = new HashSet<>();
			if (Direction.valueOf((String) o.get("direction")) == Direction.DESCENDING) {
				directions.add(SqlKind.DESCENDING);
			}
			if (NullDirection.valueOf((String) o.get("null-direction")) == NullDirection.FIRST) {
				directions.add(SqlKind.NULLS_FIRST);
			} else {
				directions.add(SqlKind.NULLS_LAST);
			}
			list.add(new RexFieldCollation(expr, directions));
		}
		return list;
	}

	private RexWindowBound toRexWindowBound(RelInput input, Map<String, Object> map) {
		if (map == null) {
			return null;
		}

		final String type = (String) map.get("type");
		switch (type) {
		case "CURRENT_ROW":
			return RexWindowBound.create(
					SqlWindow.createCurrentRow(SqlParserPos.ZERO), null);
		case "UNBOUNDED_PRECEDING":
			return RexWindowBound.create(
					SqlWindow.createUnboundedPreceding(SqlParserPos.ZERO), null);
		case "UNBOUNDED_FOLLOWING":
			return RexWindowBound.create(
					SqlWindow.createUnboundedFollowing(SqlParserPos.ZERO), null);
		case "PRECEDING":
			RexNode precedingOffset = toRex(input, map.get("offset"));
			return RexWindowBound.create(null,
					input.getCluster().getRexBuilder().makeCall(
							SqlWindow.PRECEDING_OPERATOR, precedingOffset));
		case "FOLLOWING":
			RexNode followingOffset = toRex(input, map.get("offset"));
			return RexWindowBound.create(null,
					input.getCluster().getRexBuilder().makeCall(
							SqlWindow.FOLLOWING_OPERATOR, followingOffset));
		default:
			throw new UnsupportedOperationException("cannot convert type to rex window bound " + type);
		}
	}

	private List<RexNode> toRexList(RelInput relInput, List operands) {
		final List<RexNode> list = new ArrayList<>();
		for (Object operand : operands) {
			list.add(toRex(relInput, operand));
		}
		return list;
	}

	SqlOperator toOp(Map<String, Object> map) {
		// in case different operator has the same kind, check with both name and kind.
		String name = map.get("name").toString();
		String kind = map.get("kind").toString();
		String syntax = map.get("syntax").toString();
		SqlKind sqlKind = SqlKind.valueOf(kind);
		SqlSyntax  sqlSyntax = SqlSyntax.valueOf(syntax);
		List<SqlOperator> operators = new ArrayList<>();
		SqlStdOperatorTable.instance().lookupOperatorOverloads(
				new SqlIdentifier(name, new SqlParserPos(0, 0)),
				null,
				sqlSyntax,
				operators,
				SqlNameMatchers.liberal());
		for (SqlOperator operator: operators) {
			if (operator.kind == sqlKind) {
				return operator;
			}
		}
		String class_ = (String) map.get("class");
		if (class_ != null) {
			return AvaticaUtils.instantiatePlugin(SqlOperator.class, class_);
		}
		return null;
	}

	SqlAggFunction toAggregation(Map<String, Object> map) {
		return (SqlAggFunction) toOp(map);
	}

	private Map toJson(SqlOperator operator) {
		// User-defined operators are not yet handled.
		Map map = jsonBuilder.map();
		map.put("name", operator.getName());
		map.put("kind", operator.kind.toString());
		map.put("syntax", operator.getSyntax().toString());
		return map;
	}
}
