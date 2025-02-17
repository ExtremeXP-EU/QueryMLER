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
package org.imsi.queryEREngine.apache.calcite.tools;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;

import org.imsi.queryEREngine.apache.calcite.interpreter.Bindables;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.rel.RelHomogeneousShuttle;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.RelShuttle;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableScan;
import org.imsi.queryEREngine.apache.calcite.rel.logical.LogicalTableScan;

/** Implementations of {@link RelRunner}. */
public class RelRunners {
	private RelRunners() {}

	/** Runs a relational expression by creating a JDBC connection. */
	public static PreparedStatement run(RelNode rel) {
		final RelShuttle shuttle = new RelHomogeneousShuttle() {
			@Override public RelNode visit(TableScan scan) {
				final RelOptTable table = scan.getTable();
				if (scan instanceof LogicalTableScan
						&& Bindables.BindableTableScan.canHandle(table)) {
					// Always replace the LogicalTableScan with BindableTableScan
					// because it's implementation does not require a "schema" as context.
					return Bindables.BindableTableScan.create(scan.getCluster(), table);
				}
				return super.visit(scan);
			}
		};
		rel = rel.accept(shuttle);
		try (Connection connection = DriverManager.getConnection("jdbc:calcite:")) {
			final RelRunner runner = connection.unwrap(RelRunner.class);
			return runner.prepare(rel);
		} catch (SQLException e) {
			throw new RuntimeException(e);
		}
	}
}
