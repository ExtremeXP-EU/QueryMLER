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
package org.imsi.queryEREngine.apache.calcite.interpreter;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.rel.core.Aggregate;
import org.imsi.queryEREngine.apache.calcite.rel.core.Collect;
import org.imsi.queryEREngine.apache.calcite.rel.core.Filter;
import org.imsi.queryEREngine.apache.calcite.rel.core.Join;
import org.imsi.queryEREngine.apache.calcite.rel.core.Match;
import org.imsi.queryEREngine.apache.calcite.rel.core.Project;
import org.imsi.queryEREngine.apache.calcite.rel.core.SetOp;
import org.imsi.queryEREngine.apache.calcite.rel.core.Sort;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableScan;
import org.imsi.queryEREngine.apache.calcite.rel.core.Uncollect;
import org.imsi.queryEREngine.apache.calcite.rel.core.Values;
import org.imsi.queryEREngine.apache.calcite.rel.core.Window;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;

import com.google.common.collect.ImmutableList;

/**
 * Helper methods for {@link Node} and implementations for core relational
 * expressions.
 */
public class Nodes {
	/** Extension to
	 * {@link Interpreter.CompilerImpl}
	 * that knows how to handle the core logical
	 * {@link org.imsi.queryEREngine.apache.calcite.rel.RelNode}s. */
	public static class CoreCompiler extends Interpreter.CompilerImpl {
		CoreCompiler(Interpreter interpreter, RelOptCluster cluster) {
			super(interpreter, cluster);
		}

		public void visit(Aggregate agg) {
			node = new AggregateNode(this, agg);
		}

		public void visit(Filter filter) {
			node = new FilterNode(this, filter);
		}

		public void visit(Project project) {
			node = new ProjectNode(this, project);
		}

		public void visit(Values value) {
			node = new ValuesNode(this, value);
		}

		public void visit(TableScan scan) {
			final ImmutableList<RexNode> filters = ImmutableList.of();
			node = TableScanNode.create(this, scan, filters, null);
		}

		public void visit(Bindables.BindableTableScan scan) {
			node = TableScanNode.create(this, scan, scan.filters, scan.projects);
		}

		public void visit(Sort sort) {
			node = new SortNode(this, sort);
		}

		public void visit(SetOp setOp) {
			node = new SetOpNode(this, setOp);
		}

		public void visit(Join join) {
			node = new JoinNode(this, join);
		}

		public void visit(Window window) {
			node = new WindowNode(this, window);
		}

		public void visit(Match match) {
			node = new MatchNode(this, match);
		}

		public void visit(Collect collect) {
			node = new CollectNode(this, collect);
		}

		public void visit(Uncollect uncollect) {
			node = new UncollectNode(this, uncollect);
		}
	}
}
