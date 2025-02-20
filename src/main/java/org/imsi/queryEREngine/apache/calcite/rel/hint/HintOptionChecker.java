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
package org.imsi.queryEREngine.apache.calcite.rel.hint;

import org.imsi.queryEREngine.apache.calcite.util.Litmus;

/**
 * A {@code HintOptionChecker} checks if a {@link RelHint}'s options are valid.
 *
 * <p>Every hint would have a validation when converting to a {@link RelHint}, the
 * validation logic is as follows:
 *
 * <ul>
 *   <li>Checks whether the hint name was already registered</li>
 *   <li>If a {@code HintOptionChecker} was registered,
 *   use it to check the hint options</li>
 * </ul>
 *
 * <p>In {@link HintStrategyTable} the option checker is used for
 * hints registration as an optional parameter.
 *
 * @see HintStrategyTable#validateHint
 */
@FunctionalInterface
public interface HintOptionChecker {
	/**
	 * Checks if the given hint is valid.
	 *
	 * <p>Always use the {@link Litmus#check(boolean, String, Object...)}
	 * to do the check. The default behavior is to log warnings for any hint error.
	 *
	 * @param hint The hint to check
	 * @param errorHandler The error handler
	 *
	 * @return True if the check passes
	 */
	boolean checkOptions(RelHint hint, Litmus errorHandler);
}
