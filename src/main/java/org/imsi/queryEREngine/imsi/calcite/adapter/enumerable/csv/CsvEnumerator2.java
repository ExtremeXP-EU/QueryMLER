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
package org.imsi.queryEREngine.imsi.calcite.adapter.enumerable.csv;

import java.io.IOException;
import java.io.Reader;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.TimeZone;
import java.util.concurrent.atomic.AtomicBoolean;

import org.imsi.queryEREngine.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.avatica.util.DateTimeUtils;
import org.apache.calcite.linq4j.Enumerator;
import org.imsi.queryEREngine.apache.calcite.rel.type.RelDataType;
import org.imsi.queryEREngine.apache.calcite.sql.type.SqlTypeName;
import org.imsi.queryEREngine.apache.calcite.util.Pair;
import org.imsi.queryEREngine.apache.calcite.util.Source;
import org.apache.commons.lang3.time.FastDateFormat;
import org.imsi.queryEREngine.imsi.er.KDebug;

import au.com.bytecode.opencsv.CSVReader;

/** Enumerator that reads from a CSV file.
 *
 * @param <E> Row type
 */
public class CsvEnumerator2<E> implements Enumerator<E> {
	
	 public static String getCallerClassName() {
		    StackTraceElement[] stElements = Thread.currentThread().getStackTrace();
		    for (int i = 1; i < stElements.length; i++) {
		      StackTraceElement ste = stElements[i];
		      if (!ste.getClassName().equals(KDebug.class.getName()) && ste.getClassName().indexOf("java.lang.Thread") != 0)
		        return ste.getClassName(); 
		    } 
		    return null;
		  }
		  
	private  CSVReader reader;
	private  String[] filterValues;
	private  AtomicBoolean cancelFlag;
	private  RowConverter<E> rowConverter;
	private  E current;
	private int position;
	private List<CsvFieldType> fieldTypes;

	private static final FastDateFormat TIME_FORMAT_DATE;
	private static final FastDateFormat TIME_FORMAT_TIME;
	private static final FastDateFormat TIME_FORMAT_TIMESTAMP;
	public List<String> fieldNames;
	  
	private Source source;
	static {
		final TimeZone gmt = TimeZone.getTimeZone("GMT");
		TIME_FORMAT_DATE = FastDateFormat.getInstance("yyyy-MM-dd", gmt);
		TIME_FORMAT_TIME = FastDateFormat.getInstance("HH:mm:ss", gmt);
		TIME_FORMAT_TIMESTAMP =
				FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss", gmt);
	}


	public CsvEnumerator2(Source source, AtomicBoolean cancelFlag,
			List<CsvFieldType> fieldTypes) {
		this(source, cancelFlag, fieldTypes, identityList(fieldTypes.size()));
	}

	public CsvEnumerator2(Source source, AtomicBoolean cancelFlag,
			List<CsvFieldType> fieldTypes, Integer[] fields) {
		//noinspection unchecked
		this(source, fieldTypes, cancelFlag,
				(RowConverter<E>) converter(fieldTypes, fields));
	}

	CsvEnumerator2(Source source, List<CsvFieldType> fieldTypes, AtomicBoolean cancelFlag, RowConverter<E> rowConverter) {
		this.cancelFlag = cancelFlag;
		this.source = source;
		this.rowConverter = rowConverter;
		this.cancelFlag = cancelFlag;
		this.fieldTypes = fieldTypes;
		try {
			this.position = 0;
			this.reader = openCsv(source);
			this.reader.readNext(); // skip header row
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}


	private static RowConverter<?> converter(List<CsvFieldType> fieldTypes,
			Integer[] fields) {
		return new ArrayRowConverter(fieldTypes, fields);
	}



	/** Deduces the names and types of a table's columns by reading the first line
	 * of a CSV file. */
	static RelDataType deduceRowType(JavaTypeFactory typeFactory, Source source,
			List<CsvFieldType> fieldTypes) {
		final List<RelDataType> types = new ArrayList<>();
		final List<String> names = new ArrayList<>();
		try (CSVReader reader = openCsv(source)) {
			String[] strings = reader.readNext();
			if (strings == null) {
				strings = new String[]{"EmptyFileHasNoColumns:boolean"};
			}
			for (String string : strings) {
				final String name;
				final CsvFieldType fieldType;
				final int colon = string.indexOf(':');
				if (colon >= 0) {

					name = string.substring(0, colon);
					String typeString = string.substring(colon + 1);
					fieldType = CsvFieldType.of(typeString);
					if (fieldType == null) {
						System.out.println("WARNING: Found unknown type: "
								+ typeString + " in file: " + source.path()
								+ " for column: " + name
								+ ". Will assume the type of column is string");
					}
				} else {
					name = string;
					fieldType = null;
				}
				final RelDataType type;
				if (fieldType == null) {
					type = typeFactory.createSqlType(SqlTypeName.VARCHAR);
				} else {
					type = fieldType.toType(typeFactory);
				}
				names.add(name);
				types.add(type);
				if (fieldTypes != null) {
					fieldTypes.add(fieldType);
				}
			}
		} catch (IOException e) {
			// ignore
		}
		if (names.isEmpty()) {
			names.add("line");
			types.add(typeFactory.createSqlType(SqlTypeName.VARCHAR));
		}
		return typeFactory.createStructType(Pair.zip(names, types));
	}

	public static CSVReader openCsv(Source source) throws IOException {
		
		final Reader fileReader = source.reader();
		char seperator = '\t';
//		if(source.toString().contains("publications") || source.toString().contains("venues"))
//			seperator = '>';
		if(source.toString().contains("people") || source.toString().contains("projects") || source.toString().contains("orga"))
		seperator = ',';
		
		return new CSVReader(fileReader, seperator, '"');
//		return new CSVReader(fileReader, ',');
	}

	@Override
	public E current() {
		return current;
	}

	public int position() {
		return position;
	}

	@Override
	public boolean moveNext() {
		try {
			outer:
				for (;;) {
					if (cancelFlag.get()) {
						return false;
					}
					final String[] strings = reader.readNext();
					if (strings == null) {
						if (reader instanceof CsvStreamReader) {
							try {
								Thread.sleep(CsvStreamReader.DEFAULT_MONITOR_DELAY);
							} catch (InterruptedException e) {
								throw new RuntimeException(e);
							}
							continue;
						}
						current = null;
						reader.close();
						return false;
					}
					if (filterValues != null) {
						for (int i = 0; i < strings.length; i++) {
							String filterValue = filterValues[i];
							if (filterValue != null) {
								if (!filterValue.equals(strings[i])) {
									continue outer;
								}
							}
						}
					}
					position += 1;

					current = rowConverter.convertRow(strings);
//					System.out.println(current);

					if (current == null) continue;
					return true;
				}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void reset() {
		throw new UnsupportedOperationException();
	}

	@Override
	public void close() {
		try {
			reader.close();
		} catch (IOException e) {
			throw new RuntimeException("Error closing CSV reader", e);
		}
	}

	/** Returns an array of integers {0, ..., n - 1}. */
	public static Integer[] identityList(int n) {
		Integer[] integers = new Integer[n];
		for (int i = 0; i < n; i++) {
			integers[i] = i;
		}
		return integers;
	}

	/** Row converter.
	 *
	 * @param <E> element type */
	abstract static class RowConverter<E> {
		abstract E convertRow(String[] rows);

		protected Object convert(CsvFieldType fieldType, String string) {

			if (fieldType == null) {
				return string;
			}
			switch (fieldType) {
			case BOOLEAN:
				if (string.length() == 0) {
					return null;
				}
				return Boolean.parseBoolean(string);
			case BYTE:
				if (string.length() == 0) {
					return null;
				}
				return Byte.parseByte(string);
			case SHORT:
				if (string.length() == 0) {
					return null;
				}
				return Short.parseShort(string);
			case INT:
				if (string.length() == 0) {
					return null;
				}
				return Integer.parseInt(string);
			case LONG:
				if (string.length() == 0) {
					return null;
				}
				return Long.parseLong(string);
			case FLOAT:
				if (string.length() == 0) {
					return null;
				}
				return Float.parseFloat(string);
			case DOUBLE:
				if (string.length() == 0) {
					return null;
				}
				return Double.parseDouble(string);
			case DATE:
				if (string.length() == 0) {
					return null;
				}
				try {
					Date date = TIME_FORMAT_DATE.parse(string);
					return (int) (date.getTime() / DateTimeUtils.MILLIS_PER_DAY);
				} catch (ParseException e) {
					return null;
				}
			case TIME:
				if (string.length() == 0) {
					return null;
				}
				try {
					Date date = TIME_FORMAT_TIME.parse(string);
					return (int) date.getTime();
				} catch (ParseException e) {
					return null;
				}
			case TIMESTAMP:
				if (string.length() == 0) {
					return null;
				}
				try {
					Date date = TIME_FORMAT_TIMESTAMP.parse(string);
					return date.getTime();
				} catch (ParseException e) {
					return null;
				}
			case STRING:
			default:
				return string;
			}
		}
	}

	/** Array row converter. */
	static class ArrayRowConverter extends RowConverter<Object[]> {
		private final CsvFieldType[] fieldTypes;
		private final Integer[] fields;
		// whether the row to convert is from a stream
		private final boolean stream;

		ArrayRowConverter(List<CsvFieldType> fieldTypes, Integer[] fields2) {
			this.fieldTypes = fieldTypes.toArray(new CsvFieldType[0]);
			this.fields = fields2;
			this.stream = false;
		}

		ArrayRowConverter(List<CsvFieldType> fieldTypes, Integer[] fields, boolean stream) {
			this.fieldTypes = fieldTypes.toArray(new CsvFieldType[0]);
			this.fields = fields;
			this.stream = stream;
		}

		@Override
		public Object[] convertRow(String[] strings) {
			return convertNormalRow(strings);

		}

		public Object[] convertNormalRow(String[] strings) {

			final Object[] objects = new Object[fields.length];
			try {
				for (int i = 0; i < fields.length; i++) {
					int field = fields[i];
					objects[i] = convert(fieldTypes[field], strings[field]);
				}
				return objects;
			}
			catch(Exception e) {
				return objects;
			}
		}


	}

	public AtomicBoolean getCancelFlag() {
		return this.cancelFlag;
	}


	public Source getSource() {
		return this.source;
	}

	public List<CsvFieldType> getFieldTypes() {
		
		return fieldTypes;
	}




}

// End CsvEnumerator.java
