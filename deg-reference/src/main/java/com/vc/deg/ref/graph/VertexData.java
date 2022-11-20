package com.vc.deg.ref.graph;

import com.koloboke.collect.map.IntFloatMap;
import com.koloboke.collect.map.hash.HashIntFloatMapFactory;
import com.koloboke.collect.map.hash.HashIntFloatMaps;
import com.vc.deg.FeatureVector;

/**
 * @author Nico Hezel
 */
public class VertexData  {
	
	private static final HashIntFloatMapFactory mapFactory = HashIntFloatMaps.getDefaultFactory().withDefaultValue(Integer.MIN_VALUE);
	
	private final int label;
	private final int internalId;
	private final FeatureVector data;
	private final IntFloatMap edges;
	
	public VertexData(int label, int internalId, FeatureVector data, int edgesPerVertex) {
		this.label = label;
		this.internalId = internalId;
		this.data = data;
		this.edges = mapFactory.newMutableMap(edgesPerVertex);
	}
	
	public VertexData(int label, int internalId, FeatureVector data, IntFloatMap edges) {
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
	
	public IntFloatMap getEdges() {
		return edges;
	}	
}
