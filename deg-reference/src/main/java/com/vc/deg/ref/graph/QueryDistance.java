package com.vc.deg.ref.graph;

import java.util.Comparator;

import com.vc.deg.FeatureVector;

/**
 * Distance between a query and a vertex.
 * Natural order is ascending by distance.
 * 
 * @author Nico Hezel
 *
 */
public class QueryDistance implements Comparable<QueryDistance> {
	
	protected final int queryId;
	protected final FeatureVector queryFeature;
	
	protected final VertexData vertex;
	
	protected final float distance;
	
	public QueryDistance(int queryId, FeatureVector queryFeature, VertexData vertex, float distance) {
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
	public int compareTo(QueryDistance o) {
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
	public static Comparator<QueryDistance> ascByIndex() {
		return Comparator.comparingInt(QueryDistance::getVertexLabel).thenComparingDouble(QueryDistance::getDistance);
	}

	/**
	 * Order in descending order using the index
	 * 
	 * @return
	 */
	public static Comparator<QueryDistance> descByIndex() {
		return ascByIndex().reversed();
	}
}