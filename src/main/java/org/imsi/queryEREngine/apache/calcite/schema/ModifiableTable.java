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
package org.imsi.queryEREngine.apache.calcite.schema;

import java.util.Collection;
import java.util.List;

import org.imsi.queryEREngine.apache.calcite.plan.RelOptCluster;
import org.imsi.queryEREngine.apache.calcite.plan.RelOptTable;
import org.imsi.queryEREngine.apache.calcite.prepare.Prepare;
import org.imsi.queryEREngine.apache.calcite.rel.RelNode;
import org.imsi.queryEREngine.apache.calcite.rel.core.TableModify;
import org.imsi.queryEREngine.apache.calcite.rex.RexNode;

/**
 * A table that can be modified.
 *
 * <p>NOTE: The current API is inefficient and experimental. It will change
 * without notice.</p>
 *
 * @see ModifiableView
 */
public interface ModifiableTable extends QueryableTable {
	/** Returns the modifiable collection.
	 * Modifying the collection will change the table's contents. */
	Collection getModifiableCollection();

	/** Creates a relational expression that modifies this table. */
	TableModify toModificationRel(
			RelOptCluster cluster,
			RelOptTable table,
			Prepare.CatalogReader catalogReader,
			RelNode child,
			TableModify.Operation operation,
			List<String> updateColumnList,
			List<RexNode> sourceExpressionList,
			boolean flattened);
}
