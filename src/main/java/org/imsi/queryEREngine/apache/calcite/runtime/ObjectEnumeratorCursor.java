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
package org.imsi.queryEREngine.apache.calcite.runtime;

import org.apache.calcite.avatica.util.PositionedCursor;
import org.apache.calcite.linq4j.Enumerator;

/**
 * Implementation of {@link org.apache.calcite.avatica.util.Cursor} on top of an
 * {@link org.apache.calcite.linq4j.Enumerator} that
 * returns an {@link Object} for each row.
 */
public class ObjectEnumeratorCursor extends PositionedCursor<Object> {
	private final Enumerator<Object> enumerator;

	/**
	 * Creates an ObjectEnumeratorCursor.
	 *
	 * @param enumerator Enumerator
	 */
	public ObjectEnumeratorCursor(Enumerator<Object> enumerator) {
		this.enumerator = enumerator;
	}

	@Override
	protected Getter createGetter(int ordinal) {
		return new ObjectGetter(ordinal);
	}

	@Override
	protected Object current() {
		return enumerator.current();
	}

	@Override
	public boolean next() {
		return enumerator.moveNext();
	}

	@Override
	public void close() {
		enumerator.close();
	}
}
