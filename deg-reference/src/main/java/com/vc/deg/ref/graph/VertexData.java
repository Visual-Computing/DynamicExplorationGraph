package com.vc.deg.ref.graph;

import java.util.HashMap;
import java.util.Map;

import com.vc.deg.FeatureVector;

/**
 * @author Nico Hezel
 */
public class VertexData  {
	
	protected final int label;
	protected final int internalId;
	protected final FeatureVector data;
	protected final Map<Integer,Float> edges;
	
	public VertexData(int label, int internalId, FeatureVector data, int edgesPerVertex) {
		this.label = label;
		this.internalId = internalId;
		this.data = data;
		this.edges = new HashMap<>(edgesPerVertex);
	}
	
	public VertexData(int label, int internalId, FeatureVector data, Map<Integer,Float> edges) {
		this.label = label;
		this.internalId = internalId;
		this.data = data;
		this.edges = edges;
	}
	
	public int getId() {
		return internalId;
	}

	public int getLabel() {
		return label;
	}

	public FeatureVector getFeature() {
		return data;
	}
	
	public Map<Integer,Float> getEdges() {
		return edges;
	}	
}
