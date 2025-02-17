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

/**
 * <h2>SQL Implicit Type Cast</h2>
 *
 * <h3>Work Flow</h3>
 *
 * <p>This package contains rules for implicit type coercion, it works during
 * the process of SQL validation. The transformation entrance are all kinds of
 * checkers. i.e.
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.AssignableOperandTypeChecker AssignableOperandTypeChecker},
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.ComparableOperandTypeChecker ComparableOperandTypeChecker}
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.CompositeOperandTypeChecker CompositeOperandTypeChecker},
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.FamilyOperandTypeChecker FamilyOperandTypeChecker},
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.SameOperandTypeChecker SameOperandTypeChecker},
 * {@link org.imsi.queryEREngine.apache.calcite.sql.type.SetopOperandTypeChecker SetopOperandTypeChecker}.
 *
 * <ul>
 *   <li>
 *     When user do validation for a {@link org.imsi.queryEREngine.apache.calcite.sql.SqlNode SqlNode},
 *     and the type coercion is turned on(default), the validator will check the
 *     operands/return types of all kinds of operators, if the validation passes through,
 *     the validator will just cache the data type (say RelDataType) for the
 *     {@link org.imsi.queryEREngine.apache.calcite.sql.SqlNode SqlNode} it has validated;</li>
 *   <li>If the validation fails, the validator will ask the
 *   {@link org.imsi.queryEREngine.apache.calcite.sql.validate.implicit.TypeCoercion TypeCoercion} about
 *   if there can make implicit type coercion, if the coercion rules passed, the
 *   {@link org.imsi.queryEREngine.apache.calcite.sql.validate.implicit.TypeCoercion TypeCoercion} component
 *   will replace the {@link org.imsi.queryEREngine.apache.calcite.sql.SqlNode SqlNode} with a casted one,
 *   (the node may be an operand of an operator or a column field of a selected row).</li>
 *   <li>{@link org.imsi.queryEREngine.apache.calcite.sql.validate.implicit.TypeCoercion TypeCoercion}
 *   will then update the inferred type for the casted node and
 *   the containing operator/row column type.</li>
 *   <li>If the coercion rule fails again, the validator will just throw
 *   the exception as is before.</li>
 * </ul>
 *
 * <p>For some cases, although the validation passes, we still need the type coercion, e.g. for
 * expression 1 &gt; '1', Calcite will just return false without type coercion, we do type coercion
 * eagerly here: the result expression would be transformed to "1 &gt; cast('1' as int)" and
 * the result would be true.
 *
 * <h3>Conversion SQL Contexts</h3>
 *
 * <p>The supported conversion contexts are:
 * <a href="https://docs.google.com/document/d/1g2RUnLXyp_LjUlO-wbblKuP5hqEu3a_2Mt2k4dh6RwU/edit?usp=sharing">Conversion Expressions</a>
 *
 * <p>Strategies for Finding Common Type:</p>
 * <ul>
 *   <li>If the operator has expected data types, just take them as the desired one. i.e. the UDF.
 *   </li>
 *   <li>If there is no expected data type but data type families are registered, try to coerce
 *   operand to the family's default data type, i.e. the STRING family will have a VARCHAR type.
 *   </li>
 *   <li>If neither expected data type nor families are specified, try to find the tightest common
 *   type of the node types, i.e. INTEGER and DOUBLE will return DOUBLE, the numeric precision
 *   does not lose for this case.</li>
 *   <li>If no tightest common type is found, try to find a wider type, i.e. STRING and INT
 *   will return int, we allow some precision loss when widening DECIMAL to fractional,
 *   or promote to STRING.</li>
 * </ul>
 *
 * <h3>Type Conversion Matrix</h3>
 *
 * <p>See <a href="https://docs.google.com/spreadsheets/d/1GhleX5h5W8-kJKh7NMJ4vtoE78pwfaZRJl88ULX_MgU/edit?usp=sharing">CalciteImplicitCasts</a>.
 */
package org.imsi.queryEREngine.apache.calcite.sql.validate.implicit;
