package org.imsi.queryEREngine.imsi.er.Utilities;

import org.imsi.queryEREngine.imsi.er.DataStructures.*;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Set;

public class QueryComparisonIterator implements Iterator<Comparison> {


    private final AbstractBlock block;
    private double executedComparisons;
    private double totalComparisons;
    private int innerLoop;
    private int innerLimit;
    private int outerLoop;
    private int outerLimit;
    private int[] queryEntities;
    private int[] entities;

    public QueryComparisonIterator(AbstractBlock block, Set<Integer> qIds) {
        this.block = block;
        totalComparisons = block.getNoOfComparisons();
        if (block instanceof BilateralBlock) {
            BilateralBlock bilBlock = (BilateralBlock) block;
            innerLoop = -1; // so that counting in function next() starts from 0
            innerLimit = bilBlock.getIndex2Entities().length - 1;
            outerLoop = 0;
            outerLimit = bilBlock.getIndex1Entities().length - 1;
        } else if (block instanceof UnilateralBlock) {
            UnilateralBlock uniBlock = (UnilateralBlock) block;
//            splitEntities(uniBlock, qIds);
            queryEntities = uniBlock.getQueryEntities();
            entities = uniBlock.getEntities();
            innerLoop = -1;
            innerLimit = uniBlock.getEntities().length-1;//entities.length - 1;
            outerLoop = 0;
            outerLimit = uniBlock.getQueryEntities().length-1; //queryEntities.length - 1;
        } else if (block instanceof DecomposedBlock) {
            innerLoop = -1;
            innerLimit = -1;
            outerLoop = -1; // so that counting in function next() starts from 0
            outerLimit = -1;
        }
    }

    public void splitEntities(UnilateralBlock uniBlock, Set<Integer> qIds) {
        entities = uniBlock.getEntities();
        queryEntities = Arrays.stream(entities).filter(qIds::contains).toArray();
//        entities = Arrays.stream(allEntities).filter(d -> !qIds.contains(d)).toArray();
        int m = queryEntities.length;
        int n = entities.length;
        totalComparisons = m * n;
        uniBlock.setQueryEntities(queryEntities);
//        System.out.println(queryEntities.length);
//        System.out.println(entities.length);
//        System.out.println(allEntities.length);
//        System.out.println();

    }

    @Override
    public boolean hasNext() {
        return executedComparisons < totalComparisons;
    }

    @Override
    public Comparison next() {
        if (totalComparisons <= executedComparisons) {
            System.err.println("All comparisons were already executed!");
            return null;
        }

        executedComparisons++;
        if (block instanceof BilateralBlock) {
            BilateralBlock bilBlock = (BilateralBlock) block;
            innerLoop++;
            if (innerLimit < innerLoop) {
                innerLoop = 0;
                outerLoop++;
                if (outerLimit < outerLoop) {
                    System.err.println("All comparisons were already executed!");
                    return null;
                }
            }

            return new Comparison(true, bilBlock.getIndex1Entities()[outerLoop], bilBlock.getIndex2Entities()[innerLoop]);
        } else if (block instanceof UnilateralBlock) {
            innerLoop++;
            if (innerLimit < innerLoop) {
                innerLoop = 0;
                outerLoop++;
                if (outerLimit < outerLoop) {
                    System.err.println("All comparisons were already executed!");
                    return null;
                }
            }
            int e1 = queryEntities[outerLoop];
            int e2 = entities[innerLoop];
            if(e1 == e2) executedComparisons--;
            return new Comparison(false, e1, e2);
        } else if (block instanceof DecomposedBlock) {
            DecomposedBlock deBlock = (DecomposedBlock) block;
            outerLoop++;
            return new Comparison(deBlock.isCleanCleanER(), deBlock.getEntities1()[outerLoop], deBlock.getEntities2()[outerLoop]);
        }

        return null;
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public boolean hasComparisons() {
        return totalComparisons != 0;
    }
}