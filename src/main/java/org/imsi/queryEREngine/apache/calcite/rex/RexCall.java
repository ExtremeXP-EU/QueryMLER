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
package org.imsi.queryEREngine.apache.calcite.rex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import javax.annotation.Nonnull;

import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataTypeFactory;
import org.imsi.queryEREngine.apache.calcite.sql.SqlKind;
import org.imsi.queryEREngine.apache.calcite.sql.SqlOperator;
import org.imsi.queryEREngine.apache.calcite.sql.SqlSyntax;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.util.Litmus;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;

/**
 * An expression formed by a call to an operator with zero or more expressions
 * as operands.
 *
 * <p>Operators may be binary, unary, functions, special syntactic constructs
 * like <code>CASE ... WHEN ... END</code>, or even internally generated
 * constructs like implicit type conversions. The syntax of the operator is
 * really irrelevant, because row-expressions (unlike
 * {@link org.imsi.queryEREngine.apache.calcite.sql.SqlNode SQL expressions})
 * do not directly represent a piece of source code.
 *
 * <p>It's not often necessary to sub-class this class. The smarts should be in
 * the operator, rather than the call. Any extra information about the call can
 * often be encoded as extra arguments. (These don't need to be hidden, because
 * no one is going to be generating source code from this tree.)</p>
 */
public class RexCall extends RexNode {
	/**
	 * Sort shorter digests first, then order by string representation.
	 * The result is designed for consistent output and better readability.
	 */
	private static final Comparator<String> OPERAND_READABILITY_COMPARATOR =
			Comparator.comparing(String::length).thenComparing(Comparator.naturalOrder());

	//~ Instance fields --------------------------------------------------------

	public final SqlOperator op;
	public final ImmutableList<RexNode> operands;
	public final RelDataType type;
	public final int nodeCount;

	/**
	 * Simple binary operators are those operators which expects operands from the same Domain.
	 *
	 * <p>Example: simple comparisions ({@code =}, {@code <}).
	 *
	 * <p>Note: it does not contain {@code IN} because that is defined on D x D^n.
	 */
	private static final Set<SqlKind> SIMPLE_BINARY_OPS;

	static {
		EnumSet<SqlKind> kinds = EnumSet.of(SqlKind.PLUS, SqlKind.MINUS, SqlKind.TIMES, SqlKind.DIVIDE);
		kinds.addAll(SqlKind.COMPARISON);
		kinds.remove(SqlKind.IN);
		SIMPLE_BINARY_OPS = Sets.immutableEnumSet(kinds);
	}

	//~ Constructors -----------------------------------------------------------

	protected RexCall(
			RelDataType type,
			SqlOperator op,
			List<? extends RexNode> operands) {
		this.type = Objects.requireNonNull(type, "type");
		this.op = Objects.requireNonNull(op, "operator");
		this.operands = ImmutableList.copyOf(operands);
		this.nodeCount = RexUtil.nodeCount(1, this.operands);
		assert op.getKind() != null : op;
		assert op.validRexOperands(operands.size(), Litmus.THROW) : this;
	}

	//~ Methods ----------------------------------------------------------------

	/**
	 * Appends call operands without parenthesis.
	 * {@link RexLiteral} might omit data type depending on the context.
	 * For instance, {@code null:BOOLEAN} vs {@code =(true, null)}.
	 * The idea here is to omit "obvious" types for readability purposes while
	 * still maintain {@link RelNode#getDigest()} contract.
	 *
	 * @see RexLiteral#computeDigest(RexDigestIncludeType)
	 * @param sb destination
	 * @return original StringBuilder for fluent API
	 */
	protected final StringBuilder appendOperands(StringBuilder sb) {
		if (operands.isEmpty()) {
			return sb;
		}
		List<String> operandDigests = new ArrayList<>(operands.size());
		for (int i = 0; i < operands.size(); i++) {
			RexNode operand = operands.get(i);
			if (!(operand instanceof RexLiteral)) {
				operandDigests.add(operand.toString());
				continue;
			}
			// Type information might be omitted in certain cases to improve readability
			// For instance, AND/OR arguments should be BOOLEAN, so
			// AND(true, null) is better than AND(true, null:BOOLEAN), and we keep the same info
			// +($0, 2) is better than +($0, 2:BIGINT). Note: if $0 has BIGINT, then 2 is expected to be
			// of BIGINT type as well.
			RexDigestIncludeType includeType = RexDigestIncludeType.OPTIONAL;
			if ((isA(SqlKind.AND) || isA(SqlKind.OR))
					&& operand.getType().getSqlTypeName() == SqlTypeName.BOOLEAN) {
				includeType = RexDigestIncludeType.NO_TYPE;
			}
			if (SIMPLE_BINARY_OPS.contains(getKind()) && operands.size() == 2) {
				RexNode otherArg = operands.get(1 - i);
				if ((!(otherArg instanceof RexLiteral)
						|| ((RexLiteral) otherArg).digestIncludesType() == RexDigestIncludeType.NO_TYPE)
						&& equalSansNullability(operand.getType(), otherArg.getType())) {
					includeType = RexDigestIncludeType.NO_TYPE;
				}
			}
			operandDigests.add(((RexLiteral) operand).computeDigest(includeType));
		}
		int totalLength = (operandDigests.size() - 1) * 2; // commas
		for (String s : operandDigests) {
			totalLength += s.length();
		}
		sb.ensureCapacity(sb.length() + totalLength);
		sortOperandsIfNeeded(sb, operands, operandDigests);
		for (int i = 0; i < operandDigests.size(); i++) {
			String op = operandDigests.get(i);
			if (i != 0) {
				sb.append(", ");
			}
			sb.append(op);
		}
		return sb;
	}

	private void sortOperandsIfNeeded(StringBuilder sb,
			List<RexNode> operands, List<String> operandDigests) {
		if (operands.isEmpty() || !needNormalize()) {
			return;
		}
		final SqlKind kind = op.getKind();
		if (SqlKind.SYMMETRICAL_SAME_ARG_TYPE.contains(kind)) {
			final RelDataType firstType = operands.get(0).getType();
			for (int i = 1; i < operands.size(); i++) {
				if (!equalSansNullability(firstType, operands.get(i).getType())) {
					// Arguments have different type, thus they must not be sorted
					return;
				}
			}
			// fall through: order arguments below
		} else if (!SqlKind.SYMMETRICAL.contains(kind)
				&& (kind == kind.reverse()
				|| !op.getName().equals(kind.sql)
				|| sb.length() < kind.sql.length() + 1
				|| sb.charAt(sb.length() - 1) != '(')) {
			// The operations have to be either symmetrical or reversible
			// Nothing matched => we skip argument sorting
			// Note: RexCall digest uses op.getName() that might be different from kind.sql
			// for certain calls. So we skip normalizing the calls that have customized op.getName()
			// We ensure the current string contains enough room for preceding kind.sql otherwise
			// we won't have an option to replace the operator to reverse it in case the operands are
			// reordered.
			return;
		}
		// $0=$1 is the same as $1=$0, so we make sure the digest is the same for them
		String oldFirstArg = operandDigests.get(0);
		operandDigests.sort(OPERAND_READABILITY_COMPARATOR);

		// When $1 > $0 is normalized, the operation needs to be flipped
		// So we sort arguments first, then flip the sign
		if (kind != kind.reverse()) {
			assert operands.size() == 2
					: "Compare operation must have 2 arguments: " + this
					+ ". Actual arguments are " + operandDigests;
			int operatorEnd = sb.length() - 1 /* ( */;
			int operatorStart = operatorEnd - op.getName().length();
			assert op.getName().contentEquals(sb.subSequence(operatorStart, operatorEnd))
			: "Operation name must precede opening brace like in <=(x, y). Actual content is "
			+ sb.subSequence(operatorStart, operatorEnd)
			+ " at position " + operatorStart + " in " + sb;

			SqlKind newKind = kind.reverse();

			// If arguments are the same, then we normalize < vs >
			// '<' == 60, '>' == 62, so we prefer <
			if (operandDigests.get(0).equals(operandDigests.get(1))) {
				if (newKind.compareTo(kind) > 0) {
					// If reverse kind is greater, then skip reversing
					return;
				}
			} else if (oldFirstArg.equals(operandDigests.get(0))) {
				// The sorting did not shuffle the operands, so we do not need to update operation name
				// in the digest
				return;
			}
			// Replace operator name in the digest
			sb.replace(operatorStart, operatorEnd, newKind.sql);
		}
	}

	/**
	 * This is a poorman's
	 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeUtil#equalSansNullability(RelDataTypeFactory, RelDataType, RelDataType)}
	 * <p>{@code SqlTypeUtil} requires {@link RelDataTypeFactory} which we haven't, so we assume that
	 * "not null" is represented in the type's digest as a trailing "NOT NULL" (case sensitive)
	 * @param a first type
	 * @param b second type
	 * @return true if the types are equal or the only difference is nullability
	 */
	private static boolean equalSansNullability(RelDataType a, RelDataType b) {
		String x = a.getFullTypeString();
		String y = b.getFullTypeString();
		if (x.length() < y.length()) {
			String c = x;
			x = y;
			y = c;
		}

		return (x.length() == y.length()
				|| x.length() == y.length() + 9 && x.endsWith(" NOT NULL"))
				&& x.startsWith(y);
	}

	protected @Nonnull String computeDigest(boolean withType) {
		final StringBuilder sb = new StringBuilder(op.getName());
		if ((operands.size() == 0)
				&& (op.getSyntax() == SqlSyntax.FUNCTION_ID)) {
			// Don't print params for empty arg list. For example, we want
			// "SYSTEM_USER", not "SYSTEM_USER()".
		} else {
			sb.append("(");
			appendOperands(sb);
			sb.append(")");
		}
		if (withType) {
			sb.append(":");

			// NOTE jvs 16-Jan-2005:  for digests, it is very important
			// to use the full type string.
			sb.append(type.getFullTypeString());
		}
		return sb.toString();
	}

	@Override public final @Nonnull String toString() {
		if (!needNormalize()) {
			// Non-normalize describe is requested
			return computeDigest(digestWithType());
		}
		// This data race is intentional
		String localDigest = digest;
		if (localDigest == null) {
			localDigest = computeDigest(digestWithType());
			digest = Objects.requireNonNull(localDigest);
		}
		return localDigest;
	}

	private boolean digestWithType() {
		return isA(SqlKind.CAST) || isA(SqlKind.NEW_SPECIFICATION);
	}

	@Override
	public <R> R accept(RexVisitor<R> visitor) {
		return visitor.visitCall(this);
	}

	@Override
	public <R, P> R accept(RexBiVisitor<R, P> visitor, P arg) {
		return visitor.visitCall(this, arg);
	}

	@Override
	public RelDataType getType() {
		return type;
	}

	@Override public boolean isAlwaysTrue() {
		// "c IS NOT NULL" occurs when we expand EXISTS.
		// This reduction allows us to convert it to a semi-join.
		switch (getKind()) {
		case IS_NOT_NULL:
			return !operands.get(0).getType().isNullable();
		case IS_NOT_TRUE:
		case IS_FALSE:
		case NOT:
			return operands.get(0).isAlwaysFalse();
		case IS_NOT_FALSE:
		case IS_TRUE:
		case CAST:
			return operands.get(0).isAlwaysTrue();
		default:
			return false;
		}
	}

	@Override public boolean isAlwaysFalse() {
		switch (getKind()) {
		case IS_NULL:
			return !operands.get(0).getType().isNullable();
		case IS_NOT_TRUE:
		case IS_FALSE:
		case NOT:
			return operands.get(0).isAlwaysTrue();
		case IS_NOT_FALSE:
		case IS_TRUE:
		case CAST:
			return operands.get(0).isAlwaysFalse();
		default:
			return false;
		}
	}

	@Override
	public SqlKind getKind() {
		return op.kind;
	}

	public List<RexNode> getOperands() {
		return operands;
	}

	public SqlOperator getOperator() {
		return op;
	}

	@Override public int nodeCount() {
		return nodeCount;
	}

	/**
	 * Creates a new call to the same operator with different operands.
	 *
	 * @param type     Return type
	 * @param operands Operands to call
	 * @return New call
	 */
	public RexCall clone(RelDataType type, List<RexNode> operands) {
		return new RexCall(type, op, operands);
	}

	@Override public boolean equals(Object obj) {
		return obj == this
				|| obj instanceof RexCall
				&& toString().equals(obj.toString());
	}

	@Override public int hashCode() {
		return toString().hashCode();
	}
}
