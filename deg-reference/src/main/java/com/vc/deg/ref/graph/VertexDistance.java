package com.vc.deg.ref.graph;

import java.util.Comparator;

import com.vc.deg.FeatureVector;

/**
 * Natural order is ascending by distance.
 * 
 * @author Nico Hezel
 *
 */
public class VertexDistance implements Comparable<VertexDistance> {
	
	protected final int queryId;
	protected final FeatureVector queryFeature;
	
	protected final VertexData vertex;
	
	protected final float distance;
	
	public VertexDistance(int queryId, FeatureVector queryFeature, VertexData vertex, float distance) {
		this.queryId = queryId;
		this.queryFeature = queryFeature;
		this.vertex = vertex;
		this.distance = distance;
	}
	
	public VertexData getVertex() {
		return vertex;
	}
	
	public int getVertexLabel() {
		return vertex.getLabel();
	}
	
	public int getVertexId() {
		return vertex.getId();
	}

	public float getDistance() {
		return distance;
	}
	
	@Override
	public String toString() {
		return "label:"+getVertexLabel()+", distance:"+distance;
	}
	
	@Override
	public int compareTo(VertexDistance o) {
		int cmp = Float.compare(getDistance(), o.getDistance());
        if (cmp == 0)
        	cmp = Integer.compare(getVertexLabel(), o.getVertexLabel());
        return cmp;
	}	
	
	/**
	 * Order in ascending order using the index
	 *
	 * @return
	 */
	public static Comparator<VertexDistance> ascByIndex() {
		return Comparator.comparingInt(VertexDistance::getVertexLabel).thenComparingDouble(VertexDistance::getDistance);
	}

	/**
	 * Order in descending order using the index
	 * 
	 * @return
	 */
	public static Comparator<VertexDistance> descByIndex() {
		return ascByIndex().reversed();
	}
}