package org.imsi.queryEREngine.imsi.er.Utilities;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Handles the transmutation of data into the arrow form so they can later be either
 * stored or transferred using Arrow Flight
 */
public class ArrowDataHandler {
    private final BufferAllocator allocator;
    private final VectorSchemaRoot pairVSR;
    private final VectorSchemaRoot dictVSR;
    private final HashMap<Integer, Object[]> data;
    private final int columnCount;

    /**
     * Constructs an ArrowDataHandler and stores the data passed to be used by addDictData.
     * Responsible for transmuting data into arrow format
     * @param data dictionary of IDs -> array of strings/features
     */
    public ArrowDataHandler(HashMap<Integer, Object[]> data){
        // Create a new RootAllocator
        this.allocator = new RootAllocator();
        this.data = data;
        this.columnCount = determineColumnCount();
        this.pairVSR = this.createPairVSR();
        this.dictVSR = this.createDictVSR(this.columnCount);
    }

    /**
     * Adds the pair of ids provided to the pairs table
     * @param id1 id of the first element
     * @param id2 id of the second element
     */
    public void addPair(int id1, int id2){
        IntVector idVector1 = (IntVector) this.pairVSR.getVector("id1");
        IntVector idVector2 = (IntVector) this.pairVSR.getVector("id2");

        // Set the values in the vectors
        idVector1.setSafe(getPairRowCount(), id1);
        idVector2.setSafe(getPairRowCount(), id2);

        // Increase the row count
        incrementPairRowCount();
    }

    /**
     * Adds the data provided in the constructor to the dictionary table
     */
    public void addDictData(){
        VectorSchemaRoot root = this.dictVSR;
        IntVector idVector = (IntVector) root.getVector("id");
        List<VarCharVector> utf8vectors = new ArrayList<>();
        // Populate string vector list
        for(int columnIndex = 0; columnIndex < this.columnCount; columnIndex++){
            utf8vectors.add((VarCharVector) root.getVector("column" + columnIndex));
        }

        data.forEach((key, value)->{
            idVector.setSafe(root.getRowCount(), key);

            for(int i = 0; i < this.columnCount; i++){
                byte[] bytes = value[i].toString().getBytes(StandardCharsets.UTF_8);
                utf8vectors.get(i).setSafe(root.getRowCount(), bytes, 0, bytes.length);
            }

            // Increase the row count
            root.setRowCount(root.getRowCount() + 1);
        });
    }

    /**
     * @return Table of possible pairs
     */
    public VectorSchemaRoot fetchPairs(){ return this.pairVSR; }
    /**
     * @return Dictionary of IDs -> Strings of features
     */
    public VectorSchemaRoot fetchDict() { return this.dictVSR; }

    private int getPairRowCount(){return this.pairVSR.getRowCount();}
    private void incrementPairRowCount(){this.pairVSR.setRowCount(this.pairVSR.getRowCount() + 1);}

    private VectorSchemaRoot createPairVSR(){

        // Create lists to store the vectors and fields
        List<FieldVector> vectors = new ArrayList<>();
        List<Field> fields = new ArrayList<>();
        // Create vectors for the desired columns
        IntVector idVector1 = new IntVector("id1", this.allocator);
        IntVector idVector2 = new IntVector("id2", this.allocator);
        // Add the vectors to the list
        vectors.add(idVector1);
        vectors.add(idVector2);
        // Create fields for the vectors
        Field idField1 = Field.nullable("id1", new ArrowType.Int(32, false));
        Field idField2 = Field.nullable("id2", new ArrowType.Int(32, false));
        // Add the fields to the list
        fields.add(idField1);
        fields.add(idField2);
        // Create a new metadata map for the schema
        Map<String, String> metadata = new HashMap<>();
        metadata.put("size", "0"); //placeholder
        // Create a new Schema with the fields and metadata
        Schema schema = new Schema(fields, metadata);
        // Create a new VectorSchemaRoot with the new schema and the vectors
        return new VectorSchemaRoot(schema, vectors, 0);
    }

    private VectorSchemaRoot createDictVSR(int columnCount){

        // A list to hold all fields
        List<Field> fields = new ArrayList<>();
        List<FieldVector> vectors = new ArrayList<>();

        // Add the id field and vector to the VSR
        Field idField = Field.nullable("id", new ArrowType.Int(32, false));
        fields.add(idField);
        IntVector idVector = new IntVector("id", this.allocator);
        vectors.add(idVector);

        // Initialize string fields and vectors
        for (int columnIndex = 0; columnIndex < columnCount; columnIndex++) {
            // Create a new field for this column
            FieldType fieldType = new FieldType(true, ArrowType.Utf8.INSTANCE, null);
            Field utf8Field = new Field("column" + columnIndex, fieldType, null);
            fields.add(utf8Field);

            // Create a new vector for this column - If we have the schema we can replace the "column" + columnIndex with the column name
            VarCharVector varCharVector = new VarCharVector("column" + columnIndex, this.allocator);
            vectors.add(varCharVector);
        }

        // Create schema
        Schema schema = new Schema(fields);

        // Create VectorSchemaRoot
        return new VectorSchemaRoot(schema, vectors, 0);
    }

    private int determineColumnCount(){
        // Get first key to determine column count
        int firstKey = this.data.keySet().iterator().next();
        Object[] firstRow = this.data.get(firstKey);
        return firstRow.length;
    }

    public void debug(){
        System.out.println(this.pairVSR.getVector("id1"));
        VarCharVector v = (VarCharVector) this.dictVSR.getVector("column1");
        System.out.println(new String(v.get(1), StandardCharsets.UTF_8));

		System.out.println(this.dictVSR.getVector("column2"));
    }
}