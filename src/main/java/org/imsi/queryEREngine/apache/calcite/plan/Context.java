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
package org.imsi.queryEREngine.apache.calcite.plan;

import org.imsi.queryEREngine.apache.calcite.schema.Wrapper;

/**
 * Provides library users a way to store data within the planner session and
 * access it within rules. Frameworks can implement their own implementation
 * of Context and pass that as part of the FrameworkConfig.
 *
 * <p>Simply implement the {@link #unwrap} method to return any sub-objects
 * that you wish to provide.
 */
public interface Context extends Wrapper {
}
