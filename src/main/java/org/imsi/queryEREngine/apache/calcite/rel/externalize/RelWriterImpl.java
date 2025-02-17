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

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.calcite.avatica.util.Spacer;
import org.apache.calcite.linq4j.Ord;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelWriter;
import org.imsi.queryEREngine.apache.calcite.rel.metadata.RelMetadataQuery;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;
import org.imsi.queryEREngine.apache.calcite.sql.SqlExplainLevel;
import org.imsi.queryEREngine.apache.calcite.util.Pair;

import com.google.common.collect.ImmutableList;

/**
 * Implementation of {@link org.imsi.queryEREngine.apache.calcite.rel.RelWriter}.
 */
public class RelWriterImpl implements RelWriter {
	//~ Instance fields --------------------------------------------------------

	protected final PrintWriter pw;
	private final SqlExplainLevel detailLevel;
	private final boolean withIdPrefix;
	protected final Spacer spacer = new Spacer();
	private final List<Pair<String, Object>> values = new ArrayList<>();

	//~ Constructors -----------------------------------------------------------

	public RelWriterImpl(PrintWriter pw) {
		this(pw, SqlExplainLevel.EXPPLAN_ATTRIBUTES, true);
	}

	public RelWriterImpl(
			PrintWriter pw, SqlExplainLevel detailLevel,
			boolean withIdPrefix) {
		this.pw = pw;
		this.detailLevel = detailLevel;
		this.withIdPrefix = withIdPrefix;
	}

	//~ Methods ----------------------------------------------------------------

	protected void explain_(RelNode rel,
			List<Pair<String, Object>> values) {
		List<RelNode> inputs = rel.getInputs();
		final RelMetadataQuery mq = rel.getCluster().getMetadataQuery();
		if (!mq.isVisibleInExplain(rel, detailLevel)) {
			// render children in place of this, at same level
			explainInputs(inputs);
			return;
		}

		StringBuilder s = new StringBuilder();
		spacer.spaces(s);
		if (withIdPrefix) {
			s.append(rel.getId()).append(":");
		}
		s.append(rel.getRelTypeName());
		if (detailLevel != SqlExplainLevel.NO_ATTRIBUTES) {
			int j = 0;
			for (Pair<String, Object> value : values) {
				if (value.right instanceof RelNode) {
					continue;
				}
				if (j++ == 0) {
					s.append("(");
				} else {
					s.append(", ");
				}
				s.append(value.left)
				.append("=[")
				.append(value.right)
				.append("]");
			}
			if (j > 0) {
				s.append(")");
			}
		}
		switch (detailLevel) {
		case ALL_ATTRIBUTES:
			s.append(": rowcount = ")
			.append(mq.getRowCount(rel))
			.append(", cumulative cost = ")
			.append(mq.getCumulativeCost(rel));
		}
		switch (detailLevel) {
		case NON_COST_ATTRIBUTES:
		case ALL_ATTRIBUTES:
			if (!withIdPrefix) {
				// If we didn't print the rel id at the start of the line, print
				// it at the end.
				s.append(", id = ").append(rel.getId());
			}
			break;
		}
		pw.println(s);
		spacer.add(2);
		explainInputs(inputs);
		spacer.subtract(2);
	}

	private void explainInputs(List<RelNode> inputs) {
		for (RelNode input : inputs) {
			input.explain(this);
		}
	}

	@Override
	public final void explain(RelNode rel, List<Pair<String, Object>> valueList) {
		try (RexNode.Closeable ignored = withRexNormalize()) {
			explain_(rel, valueList);
		}
	}

	@Override
	public SqlExplainLevel getDetailLevel() {
		return detailLevel;
	}

	@Override
	public RelWriter item(String term, Object value) {
		values.add(Pair.of(term, value));
		return this;
	}

	@Override
	public RelWriter done(RelNode node) {
		assert checkInputsPresentInExplain(node);
		final List<Pair<String, Object>> valuesCopy =
				ImmutableList.copyOf(values);
		values.clear();
		try (RexNode.Closeable ignored = withRexNormalize()) {
			explain_(node, valuesCopy);
		}
		pw.flush();
		return this;
	}

	private boolean checkInputsPresentInExplain(RelNode node) {
		int i = 0;
		if (values.size() > 0 && values.get(0).left.equals("subset")) {
			++i;
		}
		for (RelNode input : node.getInputs()) {
			assert values.get(i).right == input;
			++i;
		}
		return true;
	}

	/**
	 * Converts the collected terms and values to a string. Does not write to
	 * the parent writer.
	 */
	public String simple() {
		final StringBuilder buf = new StringBuilder("(");
		try (RexNode.Closeable ignored = withRexNormalize()) {
			for (Ord<Pair<String, Object>> ord : Ord.zip(values)) {
				if (ord.i > 0) {
					buf.append(", ");
				}
				buf.append(ord.e.left).append("=[").append(ord.e.right).append("]");
			}
			buf.append(")");
		}
		return buf.toString();
	}
}
