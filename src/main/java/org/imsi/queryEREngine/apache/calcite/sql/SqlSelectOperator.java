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
package org.imsi.queryEREngine.apache.calcite.sql;

import java.util.ArrayList;
import java.util.List;

import org.imsi.queryEREngine.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos;
import org.imsi.queryEREngine.apache.calcite.sql.type.ReturnTypes;
import org.imsi.queryEREngine.apache.calcite.sql.util.SqlBasicVisitor;
import org.imsi.queryEREngine.apache.calcite.sql.util.SqlVisitor;

/**
 * An operator describing a query. (Not a query itself.)
 *
 * <p>Operands are:</p>
 *
 * <ul>
 * <li>0: distinct ({@link SqlLiteral})</li>
 * <li>1: selectClause ({@link SqlNodeList})</li>
 * <li>2: fromClause ({@link SqlCall} to "join" operator)</li>
 * <li>3: whereClause ({@link SqlNode})</li>
 * <li>4: havingClause ({@link SqlNode})</li>
 * <li>5: groupClause ({@link SqlNode})</li>
 * <li>6: windowClause ({@link SqlNodeList})</li>
 * <li>7: orderClause ({@link SqlNode})</li>
 * </ul>
 */
public class SqlSelectOperator extends SqlOperator {
	public static final SqlSelectOperator INSTANCE =
			new SqlSelectOperator();

	//~ Constructors -----------------------------------------------------------

	private SqlSelectOperator() {
		super("SELECT", SqlKind.SELECT, 2, true, ReturnTypes.SCOPE, null, null);
	}

	//~ Methods ----------------------------------------------------------------

	@Override
	public SqlSyntax getSyntax() {
		return SqlSyntax.SPECIAL;
	}

	@Override
	public SqlCall createCall(
			SqlLiteral functionQualifier,
			SqlParserPos pos,
			SqlNode... operands) {
		assert functionQualifier == null;
		return new SqlSelect(pos,
				(SqlNodeList) operands[0],
				(SqlNodeList) operands[1],
				operands[2],
				operands[3],
				(SqlNodeList) operands[4],
				operands[5],
				(SqlNodeList) operands[6],
				(SqlNodeList) operands[7],
				operands[8],
				operands[9],
				(SqlNodeList) operands[10]);
	}

	/**
	 * Creates a call to the <code>SELECT</code> operator.
	 *
	 * @param keywordList List of keywords such DISTINCT and ALL, or null
	 * @param selectList  The SELECT clause, or null if empty
	 * @param fromClause  The FROM clause
	 * @param whereClause The WHERE clause, or null if not present
	 * @param groupBy     The GROUP BY clause, or null if not present
	 * @param having      The HAVING clause, or null if not present
	 * @param windowDecls The WINDOW clause, or null if not present
	 * @param orderBy     The ORDER BY clause, or null if not present
	 * @param offset      Expression for number of rows to discard before
	 *                    returning first row
	 * @param fetch       Expression for number of rows to fetch
	 * @param pos         The parser position, or
	 *                    {@link org.imsi.queryEREngine.apache.calcite.sql.parser.SqlParserPos#ZERO}
	 *                    if not specified; must not be null.
	 * @return A {@link SqlSelect}, never null
	 */
	public SqlSelect createCall(
			SqlNodeList keywordList,
			SqlNodeList selectList,
			SqlNode fromClause,
			SqlNode whereClause,
			SqlNodeList groupBy,
			SqlNode having,
			SqlNodeList windowDecls,
			SqlNodeList orderBy,
			SqlNode offset,
			SqlNode fetch,
			SqlNodeList hints,
			SqlParserPos pos) {
		return new SqlSelect(
				pos,
				keywordList,
				selectList,
				fromClause,
				whereClause,
				groupBy,
				having,
				windowDecls,
				orderBy,
				offset,
				fetch,
				hints);
	}

	@Override
	public <R> void acceptCall(
			SqlVisitor<R> visitor,
			SqlCall call,
			boolean onlyExpressions,
			SqlBasicVisitor.ArgHandler<R> argHandler) {
		if (!onlyExpressions) {
			// None of the arguments to the SELECT operator are expressions.
			super.acceptCall(visitor, call, onlyExpressions, argHandler);
		}
	}

	@Override
	@SuppressWarnings("deprecation")
	public void unparse(
			SqlWriter writer,
			SqlCall call,
			int leftPrec,
			int rightPrec) {
		SqlSelect select = (SqlSelect) call;
		final SqlWriter.Frame selectFrame =
				writer.startList(SqlWriter.FrameTypeEnum.SELECT);
		writer.sep("SELECT");

		if (select.hasHints()) {
			writer.sep("/*+");
			select.hints.unparse(writer, leftPrec, rightPrec);
			writer.print("*/");
			writer.newlineAndIndent();
		}

		for (int i = 0; i < select.keywordList.size(); i++) {
			final SqlNode keyword = select.keywordList.get(i);
			keyword.unparse(writer, 0, 0);
		}
		writer.topN(select.fetch, select.offset);
		final SqlNodeList selectClause =
				select.selectList != null
				? select.selectList
						: SqlNodeList.of(SqlIdentifier.star(SqlParserPos.ZERO));
		writer.list(SqlWriter.FrameTypeEnum.SELECT_LIST, SqlWriter.COMMA,
				selectClause);

		if (select.from != null) {
			// Calcite SQL requires FROM but MySQL does not.
			writer.sep("FROM");

			// for FROM clause, use precedence just below join operator to make
			// sure that an un-joined nested select will be properly
			// parenthesized
			final SqlWriter.Frame fromFrame =
					writer.startList(SqlWriter.FrameTypeEnum.FROM_LIST);
			select.from.unparse(
					writer,
					SqlJoin.OPERATOR.getLeftPrec() - 1,
					SqlJoin.OPERATOR.getRightPrec() - 1);
			writer.endList(fromFrame);
		}

		if (select.where != null) {
			writer.sep("WHERE");

			if (!writer.isAlwaysUseParentheses()) {
				SqlNode node = select.where;

				// decide whether to split on ORs or ANDs
				SqlBinaryOperator whereSep = SqlStdOperatorTable.AND;
				if ((node instanceof SqlCall)
						&& node.getKind() == SqlKind.OR) {
					whereSep = SqlStdOperatorTable.OR;
				}

				// unroll whereClause
				final List<SqlNode> list = new ArrayList<>(0);
				while (node.getKind() == whereSep.kind) {
					assert node instanceof SqlCall;
					final SqlCall call1 = (SqlCall) node;
					list.add(0, call1.operand(1));
					node = call1.operand(0);
				}
				list.add(0, node);

				// unparse in a WHERE_LIST frame
				writer.list(SqlWriter.FrameTypeEnum.WHERE_LIST, whereSep,
						new SqlNodeList(list, select.where.getParserPosition()));
			} else {
				select.where.unparse(writer, 0, 0);
			}
		}
		if (select.groupBy != null) {
			writer.sep("GROUP BY");
			final SqlNodeList groupBy =
					select.groupBy.size() == 0 ? SqlNodeList.SINGLETON_EMPTY
							: select.groupBy;
			writer.list(SqlWriter.FrameTypeEnum.GROUP_BY_LIST, SqlWriter.COMMA,
					groupBy);
		}
		if (select.having != null) {
			writer.sep("HAVING");
			select.having.unparse(writer, 0, 0);
		}
		if (select.windowDecls.size() > 0) {
			writer.sep("WINDOW");
			writer.list(SqlWriter.FrameTypeEnum.WINDOW_DECL_LIST, SqlWriter.COMMA,
					select.windowDecls);
		}
		if (select.orderBy != null && select.orderBy.size() > 0) {
			writer.sep("ORDER BY");
			writer.list(SqlWriter.FrameTypeEnum.ORDER_BY_LIST, SqlWriter.COMMA,
					select.orderBy);
		}
		writer.fetchOffset(select.fetch, select.offset);
		writer.endList(selectFrame);
	}

	@Override
	public boolean argumentMustBeScalar(int ordinal) {
		return ordinal == SqlSelect.WHERE_OPERAND;
	}
}
